import logging
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Categorical
from torch.utils.data import Subset
from .strategy import Strategy
import mlflow
import pdb
from vendi_score import image_utils
from scipy import stats


class CombineSampling(Strategy):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args, train_dataset_list):

        super(CombineSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)

        self.stream_buffer_size = 50000
        self.hook_norm = np.array([])
        self.queried = []
        self.cycle = 0
        self.labeled_dataset_list = []
        self.train_dataset_list = train_dataset_list

    # backwardhook function
    def printgradnorm(self, module, _, grad_output):
        # パラメータ数
        for i in module.parameters():
            self.hook_norm = np.append(self.hook_norm, (grad_output[0].norm().item())/i.numel())
            break
    
    def get_vendi_score(self, idxs):
        queried_list = []
        for i in idxs:
            queried_list.append(self.train_dataset_list[i])
        for data in self.labeled_dataset_list:
            queried_list.append(data)
        evs = image_utils.embedding_vendi_score(queried_list, device=self.device)
        return evs

    def get_queried_vendi_score(self, idxs):
        queried_list = []
        for i in idxs:
            queried_list.append(self.train_dataset_list[i])
        evs = image_utils.embedding_vendi_score(queried_list, device=self.device)
        return evs

    def compute_gradnorm(self, loss):
        grad_norm = torch.tensor([]).to(torch.device('cuda'))
        gradnorm = 0.0
        self.model.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            if param.grad is not None:
                gradnorm = torch.norm(param.grad)
                gradnorm = gradnorm.unsqueeze(0)
                grad_norm = torch.cat((grad_norm, gradnorm), 0)
        return grad_norm

    def get_uncertainty(self, unlabeled_dataset, mode):

        hook_norm = np.array([])
        grad_norm = torch.tensor([]).to(torch.device('cuda'))
        uncertainty = torch.tensor([]).to(torch.device('cuda'))
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, pin_memory=True)
        criterion = nn.CrossEntropyLoss()
        
        loop = tqdm(unlabeled_loader, unit='batch', desc='| Calculate Score |', dynamic_ncols=True)
        for (inputs) in loop:
            self.hook_norm = np.array([])
            inputs = inputs.to(torch.device('cuda'))

            outputs = self.model(inputs)
            posterior = F.softmax(outputs, dim=1)

            loss = 0.0

            if mode == 0:   # expected-gradnorm
                posterior = posterior.squeeze()
                for i in range(self.target_classes):
                    label = torch.full([1], i)
                    label = label.to(torch.device('cuda'))
                    loss += posterior[i] * criterion(outputs, label)

            if mode == 1:  # entropy-gradnorm
                loss = Categorical(probs=posterior).entropy()

            pred_gradnorm = self.compute_gradnorm(loss)
            if len(hook_norm) == 0:
                hook_norm = self.hook_norm
                grad_norm = pred_gradnorm
            else:
                hook_norm += self.hook_norm
                grad_norm += pred_gradnorm

            pred_gradnorm = torch.sum(pred_gradnorm)
            pred_gradnorm = pred_gradnorm.unsqueeze(0)

            uncertainty = torch.cat((uncertainty, pred_gradnorm), 0)
            
        logging.info(f'all_gradnorm: {grad_norm}')

        return torch.sum(uncertainty).cpu()

    def furthest_first(self, unlabeled_embeddings, labeled_embeddings, n):
        
        unlabeled_embeddings = unlabeled_embeddings.to(self.device)
        labeled_embeddings = labeled_embeddings.to(self.device)
        
        dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
        min_dist = torch.min(dist_ctr, dim=1)[0]
                
        idxs = []
        for i in range(n):
            idx = torch.argmax(min_dist)
            idxs.append(idx.item())
            dist_new_ctr = torch.cdist(unlabeled_embeddings, unlabeled_embeddings[[idx],:])

            min_dist = torch.minimum(min_dist, dist_new_ctr[:,0])

        return idxs

    def init_centers(self, X, K, device):
        pdist = nn.PairwiseDistance(p=2)
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        #print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
                D2 = torch.flatten(D2)
                D2 = D2.cpu().numpy().astype(float)
            else:
                newD = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
                newD = torch.flatten(newD)
                newD = newD.cpu().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]

            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        #gram = np.matmul(X[indsAll], X[indsAll].T)
        #val, _ = np.linalg.eig(gram)
        #val = np.abs(val)
        #vgt = val[val > 1e-2]
        return indsAll

    def select(self, budget):
        self.cycle += 1
        self.model.eval()

        # set backward_hook
        # self.model.conv1.register_backward_hook(self.printgradnorm)
        # self.model.layer1[1].conv2.register_backward_hook(self.printgradnorm)
        # self.model.layer2[1].conv2.register_backward_hook(self.printgradnorm)
        # self.model.layer3[1].conv2.register_backward_hook(self.printgradnorm)
        # self.model.layer4[1].conv2.register_backward_hook(self.printgradnorm)
        # self.model.fc.register_backward_hook(self.printgradnorm)
        
        # random selection
        random_idxs = np.random.permutation(len(self.unlabeled_dataset))[:budget]
        random_idxs = random_idxs.tolist()

        # Uncetainty Sampling
        # Calculate probability
        buffered_stream = Subset(self.unlabeled_dataset, list(range( min(len(self.unlabeled_dataset), self.stream_buffer_size))))
        probs = self.predict_prob(buffered_stream)
        probs_sorted, _ = probs.sort(descending=False)

        # margin selection
        # print('margin sampling')
        # margin_idxs = []
        # batch_scores = probs_sorted[:, 1] - probs_sorted[:, 0]
        # batch_scores = [(x, i) for i,x in enumerate(batch_scores)]
        # batch_scores = sorted(batch_scores, key=lambda x: x[0], reverse=True)
        # # HACK: MAX Margin
        # # batch_scores = sorted(batch_scores, key=lambda x: x[0], reverse=False)
        # for i in range(budget):
        #     margin_idxs.append(batch_scores[i][1])

        # entropy selection
        print('entropy sampling')
        entropy_idxs = []
        log_probs = torch.log(probs)
        batch_scores = -(probs*log_probs).sum(1)
        batch_scores = [(x, i) for i,x in enumerate(batch_scores)]
        batch_scores = sorted(batch_scores, key=lambda x: x[0], reverse=True)
        # HACK: MIN Entropy
        # batch_scores = sorted(batch_scores, key=lambda x: x[0], reverse=False)
        for i in range(budget):
            entropy_idxs.append(batch_scores[i][1])

        # leastconf selection
        # print('leastconf sampling')
        # leastconf_idxs = []
        # batch_scores = -probs.max(1)[0]
        # batch_scores = [(x, i) for i,x in enumerate(batch_scores)]
        # batch_scores = sorted(batch_scores, key=lambda x: x[0], reverse=True)
        # # HACK: MAX Confidence
        # # batch_scores = sorted(batch_scores, key=lambda x: x[0], reverse=False)
        # for i in range(budget):
        #     leastconf_idxs.append(batch_scores[i][1])
        
        del buffered_stream, probs, probs_sorted, batch_scores, log_probs

        # Diversity Sampling
        embeddings = self.get_embedding(self.unlabeled_dataset)
        embeddings = embeddings.to('cpu').detach().numpy().copy()

        # kmeans selection
        # print('kmeans sampling')
        # cluster_learner = KMeans(n_clusters=budget)
        # cluster_learner.fit(embeddings)
        # cluster_idxs = cluster_learner.predict(embeddings)
        # centers = cluster_learner.cluster_centers_[cluster_idxs]
        # dis = (embeddings - centers)**2
        # dis = dis.sum(axis=1)
        # kmeans_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(budget)])

        # core-set selection
        print('coreset sampling')
        class NoLabelDataset(Dataset):
            def __init__(self, wrapped_dataset):
                self.wrapped_dataset = wrapped_dataset
            def __getitem__(self, index):
                features, _ = self.wrapped_dataset[index]
                return features
            def __len__(self):
                return len(self.wrapped_dataset)
        embedding_labeled = self.get_embedding(NoLabelDataset(self.labeled_dataset))
        embeddings = torch.from_numpy(embeddings.astype(np.float32)).clone()
        coreset_idxs = self.furthest_first(embeddings, embedding_labeled, budget)

        # badge selfction
        print('BADGE smapling')
        gradEmbedding = self.get_grad_embedding(self.unlabeled_dataset, True, "fc")
        badge_idxs = self.init_centers(gradEmbedding.cpu().numpy(), budget, self.device)


        # Calculate expected grad-norm of each query set
        grad_norm = np.array([])
        self.evs = []
        self.q_evs = []

        labeled_dataset = Subset(self.unlabeled_dataset, random_idxs)
        evs = self.get_vendi_score(random_idxs)
        q_evs = self.get_queried_vendi_score(random_idxs)
        self.evs.append(evs)
        self.q_evs.append(q_evs)
        grad_norm = np.append(grad_norm, self.get_uncertainty(labeled_dataset, mode=0).item())
        mlflow.log_metric('random_score', grad_norm[-1], step=self.cycle)
        mlflow.log_metric('random evs', evs, step=self.cycle)
        mlflow.log_metric('random q_evs', q_evs, step=self.cycle)
        
        # labeled_dataset = Subset(self.unlabeled_dataset, margin_idxs)
        # evs = self.get_vendi_score(margin_idxs)
        # self.evs.append(evs)
        # grad_norm = np.append(grad_norm, self.get_uncertainty(labeled_dataset, mode=0).item())
        # mlflow.log_metric('margin_score', grad_norm[-1], step=self.cycle)
        # mlflow.log_metric('margin evs', evs, step=self.cycle)

        labeled_dataset = Subset(self.unlabeled_dataset, entropy_idxs)
        evs = self.get_vendi_score(entropy_idxs)
        q_evs = self.get_queried_vendi_score(entropy_idxs)
        self.evs.append(evs)
        self.q_evs.append(q_evs)
        grad_norm = np.append(grad_norm, self.get_uncertainty(labeled_dataset, mode=0).item())
        mlflow.log_metric('entropy_score', grad_norm[-1], step=self.cycle)
        mlflow.log_metric('entropy evs', evs, step=self.cycle)
        mlflow.log_metric('entropy q_evs', q_evs, step=self.cycle)

        # labeled_dataset = Subset(self.unlabeled_dataset, leastconf_idxs)
        # evs = self.get_vendi_score(leastconf_idxs)
        # self.evs.append(evs)
        # grad_norm = np.append(grad_norm, self.get_uncertainty(labeled_dataset, mode=0).item())
        # mlflow.log_metric('leastconf_score', grad_norm[-1], step=self.cycle)
        # mlflow.log_metric('leastconf evs', evs, step=self.cycle)

        # labeled_dataset = Subset(self.unlabeled_dataset, kmeans_idxs)
        # evs = self.get_vendi_score(kmeans_idxs)
        # self.evs.append(evs)
        # grad_norm = np.append(grad_norm, self.get_uncertainty(labeled_dataset, mode=0).item())
        # mlflow.log_metric('kmeans_score', grad_norm[-1], step=self.cycle)
        # mlflow.log_metric('kmeans evs', evs, step=self.cycle)

        labeled_dataset = Subset(self.unlabeled_dataset, coreset_idxs)
        evs = self.get_vendi_score(coreset_idxs)
        q_evs = self.get_queried_vendi_score(coreset_idxs)
        self.evs.append(evs)
        self.q_evs.append(q_evs)
        grad_norm = np.append(grad_norm, self.get_uncertainty(labeled_dataset, mode=0).item())
        mlflow.log_metric('coreset_score', grad_norm[-1], step=self.cycle)
        mlflow.log_metric('coreset evs', evs, step=self.cycle)
        mlflow.log_metric('coreset q_evs', q_evs, step=self.cycle)

        labeled_dataset = Subset(self.unlabeled_dataset, badge_idxs)
        evs = self.get_vendi_score(badge_idxs)
        q_evs = self.get_queried_vendi_score(badge_idxs)
        self.evs.append(evs)
        self.q_evs.append(q_evs)
        grad_norm = np.append(grad_norm, self.get_uncertainty(labeled_dataset, mode=0).item())
        mlflow.log_metric('badge_score', grad_norm[-1], step=self.cycle)
        mlflow.log_metric('badge evs', evs, step=self.cycle)
        mlflow.log_metric('badge q_evs', q_evs, step=self.cycle)

        logging.info(f'grad_norm: {grad_norm}')
        logging.info(f'evs: {self.evs}')
        logging.info(f'q_evs: {self.q_evs}')
        normalized_grad = [(i-min(grad_norm))/(max(grad_norm)-min(grad_norm)) for i in grad_norm]
        normalized_relative_grad = [i/sum(normalized_grad) for i in normalized_grad]
        rank_grad = np.argsort(np.argsort(grad_norm))
        normalized_evs = [(i-min(self.evs))/(max(self.evs)-min(self.evs)) for i in self.evs]
        normalized_relative_evs = [i/sum(normalized_evs) for i in normalized_evs]
        rank_evs = np.argsort(np.argsort(np.array(self.evs)))
        normalized_qevs = [(i-min(self.q_evs))/(max(self.q_evs)-min(self.q_evs)) for i in self.q_evs]
        normalized_relative_qevs = [i/sum(normalized_qevs) for i in normalized_qevs]
        rank_qevs = np.argsort(np.argsort(np.array(self.q_evs)))
        logging.info(f'normalized grad = {normalized_grad}')
        logging.info(f'normalized relative grad = {normalized_relative_grad}')
        logging.info(f'normalized evs = {normalized_evs}')
        logging.info(f'normalized relative evs = {normalized_relative_evs}')
        logging.info(f'normalized qevs = {normalized_qevs}')
        logging.info(f'normalized relative qevs = {normalized_relative_qevs}')
        logging.info(f'rank: {[rank_grad[i]+rank_evs[i] for i in range(len(rank_grad))]}')
        logging.info(f'q_rank: {[rank_grad[i]+rank_qevs[i] for i in range(len(rank_grad))]}')
        
        normalized_score = list(map(lambda x, y:x+y, normalized_grad, normalized_evs))
        normalized_relative_score = list(map(lambda x, y:x+y, normalized_relative_grad, normalized_relative_evs))
        rank = list(map(lambda x, y:x+y, rank_grad, rank_evs))
        q_normalized_score = list(map(lambda x, y:x+y, normalized_grad, normalized_qevs))
        q_normalized_relative_score = list(map(lambda x, y:x+y, normalized_relative_grad, normalized_relative_qevs))
        rank_q = list(map(lambda x, y:x+y, rank_grad, rank_qevs))
        logging.info(f'normalized_grad_norm+normalized_evs = {normalized_score}')
        logging.info(f'q_normalized_grad_norm+q_normalized_evs = {q_normalized_score}')
        mlflow.log_metric('normalized random score', normalized_score[0], step=self.cycle)
        mlflow.log_metric('normalized entropy score', normalized_score[1], step=self.cycle)
        mlflow.log_metric('normalized coreset score', normalized_score[2], step=self.cycle)
        mlflow.log_metric('normalized badge score', normalized_score[3], step=self.cycle)
        mlflow.log_metric('normalized random relative score', normalized_relative_score[0], step=self.cycle)
        mlflow.log_metric('normalized entropy relative score', normalized_relative_score[1], step=self.cycle)
        mlflow.log_metric('normalized coreset relative score', normalized_relative_score[2], step=self.cycle)
        mlflow.log_metric('normalized badge relative score', normalized_relative_score[3], step=self.cycle)
        mlflow.log_metric('rank random score', rank[0], step=self.cycle)
        mlflow.log_metric('rank entropy score', rank[1], step=self.cycle)
        mlflow.log_metric('rank coreset score', rank[2], step=self.cycle)
        mlflow.log_metric('rank badge score', rank[3], step=self.cycle)
        mlflow.log_metric('q_normalized random score', q_normalized_score[0], step=self.cycle)
        mlflow.log_metric('q_normalized entropy score', q_normalized_score[1], step=self.cycle)
        mlflow.log_metric('q_normalized coreset score', q_normalized_score[2], step=self.cycle)
        mlflow.log_metric('q_normalized badge score', q_normalized_score[3], step=self.cycle)
        mlflow.log_metric('q_normalized random relative score', q_normalized_relative_score[0], step=self.cycle)
        mlflow.log_metric('q_normalized entropy relative score', q_normalized_relative_score[1], step=self.cycle)
        mlflow.log_metric('q_normalized coreset relative score', q_normalized_relative_score[2], step=self.cycle)
        mlflow.log_metric('q_normalized badge relative score', q_normalized_relative_score[3], step=self.cycle)
        mlflow.log_metric('rank_q random score', rank_q[0], step=self.cycle)
        mlflow.log_metric('rank_q entropy score', rank_q[1], step=self.cycle)
        mlflow.log_metric('rank_q coreset score', rank_q[2], step=self.cycle)
        mlflow.log_metric('rank_q badge score', rank_q[3], step=self.cycle)
        
        # HACK 
        best_strat = np.argmax(normalized_score)
        # best_strat = np.argmax(rank)
        # best_strat = np.argmax(rank_q)
        # best_strat = np.argmax(q_normalized_score)
        if best_strat == 0:
            logging.info('queried by Random')
            for i in random_idxs:
                self.labeled_dataset_list.append(self.train_dataset_list[i])
            self.train_dataset_list = np.delete(self.train_dataset_list, random_idxs)
            return random_idxs
        elif best_strat == 1:
            logging.info('queried by Entropy')
            for i in entropy_idxs:
                self.labeled_dataset_list.append(self.train_dataset_list[i])
            self.train_dataset_list = np.delete(self.train_dataset_list, entropy_idxs)
            return entropy_idxs
        elif best_strat == 2:
            logging.info('queried by Coreset')
            for i in coreset_idxs:
                self.labeled_dataset_list.append(self.train_dataset_list[i])
            self.train_dataset_list = np.delete(self.train_dataset_list, coreset_idxs)
            return coreset_idxs
        elif best_strat == 3:
            logging.info('queried by BADGE')
            for i in coreset_idxs:
                self.labeled_dataset_list.append(self.train_dataset_list[i])
            self.train_dataset_list = np.delete(self.train_dataset_list, coreset_idxs)
            return coreset_idxs

        # elif best_strat == 3:
        #     logging.info('queried by leastconf')
        #     for i in leastconf_idxs:
        #         self.labeled_dataset_list.append(self.train_dataset_list[i])
        #     self.train_dataset_list = np.delete(self.train_dataset_list, leastconf_idxs)
        #     return leastconf_idxs
        # elif best_strat == 4:
        #     logging.info('queried by Kmeans')
        #     for i in kmeans_idxs:
        #         self.labeled_dataset_list.append(self.train_dataset_list[i])
        #     self.train_dataset_list = np.delete(self.train_dataset_list, kmeans_idxs)
        #     return kmeans_idxs
        # elif best_strat == 1:
        #     logging.info('queied by Margin')
        #     for i in margin_idxs:
        #         self.labeled_dataset_list.append(self.train_dataset_list[i])
        #     self.train_dataset_list = np.delete(self.train_dataset_list, margin_idxs)
        #     return margin_idxs