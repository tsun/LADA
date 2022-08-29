import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from .utils import ActualSequentialSampler
from .sampler import register_strategy, SamplingStrategy

@register_strategy('LADA')
class LADASampling(SamplingStrategy):
    '''
    Implement Local context-aware sampling (LAS)
    '''

    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(LADASampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)

    def query(self, n, epoch):
        idxs_unlabeled = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.query_dset, sampler=train_sampler,
                                                  num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                                                  batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
        # build nearest neighbors
        self.model.eval()
        all_probs = []
        all_embs = []
        with torch.no_grad():
            for batch_idx, (data, target, _, *_) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                scores, embs = self.model(data, with_emb=True)
                all_embs.append(embs.cpu())
                probs = F.softmax(scores, dim=-1)
                all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs)
        all_embs = F.normalize(torch.cat(all_embs), dim=-1)

        # get Q_score
        sim = all_embs.cpu().mm(all_embs.transpose(1, 0))
        K = self.cfg.LADA.S_K
        sim_topk, topk = torch.topk(sim, k=K + 1, dim=1)
        sim_topk, topk = sim_topk[:, 1:], topk[:, 1:]
        wgt_topk = (sim_topk / sim_topk.sum(dim=1, keepdim=True))

        Q_score = -((all_probs[topk] * all_probs.unsqueeze(1)).sum(-1) * wgt_topk).sum(-1)

        # propagate Q_score
        for i in range(self.cfg.LADA.S_PROP_ITER):
            Q_score += (wgt_topk * Q_score[topk]).sum(-1) * self.cfg.LADA.S_PROP_COEF

        m_idxs = Q_score.sort(descending=True)[1]

        # oversample and find centroids
        Kc = self.cfg.LADA.S_Kc
        m_topk = m_idxs[:n * (1 + Kc)]
        km = KMeans(n_clusters=n)
        km.fit(all_embs[m_topk])
        dists = euclidean_distances(km.cluster_centers_, all_embs[m_topk])
        sort_idxs = dists.argsort(axis=1)
        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n - len(q_idxs)
            ax += 1

        q_idxs = m_idxs[q_idxs].cpu().numpy()
        self.query_dset.rand_transform = None

        return idxs_unlabeled[q_idxs]

