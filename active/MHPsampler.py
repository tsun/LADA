import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from .utils import ActualSequentialSampler
from .sampler import register_strategy, SamplingStrategy


@register_strategy('MHP')
class MHPSampling(SamplingStrategy):
    '''
    Implements MHPL: Minimum Happy Points Learning for Active Source Free Domain Adaptation (CVPR'23)
    '''

    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(MHPSampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)

    def query(self, n, epoch):
        idxs_unlabeled = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.query_dset, sampler=train_sampler,
                                                  num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                                                  batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
        self.model.eval()
        all_probs = []
        all_embs = []
        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                scores, embs = self.model(data, with_emb=True)
                all_embs.append(embs.cpu())
                probs = F.softmax(scores, dim=-1)
                all_probs.append(probs)

        all_probs = torch.cat(all_probs)
        all_embs = F.normalize(torch.cat(all_embs), dim=-1)

        # find KNN
        sim = all_embs.cpu().mm(all_embs.transpose(1, 0))
        K = self.cfg.LADA.S_K
        sim_topk, topk = torch.topk(sim, k=K + 1, dim=1)
        sim_topk, topk = sim_topk[:, 1:], topk[:, 1:]

        # get NP scores
        all_preds = all_probs.argmax(-1)
        Sp = (torch.eye(self.num_classes)[all_preds[topk]]).sum(1)
        Sp = Sp / Sp.sum(-1, keepdim=True)
        NP = -(torch.log(Sp+1e-9)*Sp).sum(-1)

        # get NA scores
        NA = sim_topk.sum(-1) / K
        NAU = NP*NA
        sort_idxs = NAU.argsort(descending=True)

        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            if topk[sort_idxs[ax]][0] not in q_idxs:
                q_idxs.append(sort_idxs[ax])
            rem = n - len(q_idxs)
            ax += 1

        q_idxs = np.array(q_idxs)

        return idxs_unlabeled[q_idxs]

