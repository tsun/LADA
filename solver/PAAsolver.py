import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from dataset.image_list import ImageList
from dataset.transform import rand_transform
from .solver import BaseSolver, register_solver


@register_solver('RAA')
class RAASolver(BaseSolver):
    """
      Implement Random Anchor set Augmentation (RAA)
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt,
                 ada_stage, device, cfg, **kwargs):
        super(RAASolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader,
                                         joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs)

    def solve(self, epoch, seq_query_loader):
        K = self.cfg.LADA.A_K
        th = self.cfg.LADA.A_TH

        # create an anchor set
        if len(self.tgt_sup_loader) > 0:
            tgt_sup_dataset = self.tgt_sup_loader.dataset
            tgt_sup_samples = [tgt_sup_dataset.samples[i] for i in self.tgt_sup_loader.sampler.indices]
            seed_dataset = ImageList(tgt_sup_samples, root=tgt_sup_dataset.root, transform=tgt_sup_dataset.transform)
            seed_dataset.rand_transform = rand_transform
            seed_dataset.rand_num = self.cfg.LADA.A_RAND_NUM
            seed_loader = torch.utils.data.DataLoader(seed_dataset, shuffle=True,
                                          batch_size=self.tgt_sup_loader.batch_size, num_workers=self.tgt_sup_loader.num_workers)
            seed_idxs = self.tgt_sup_loader.sampler.indices.tolist()
            seed_iter = iter(seed_loader)
            seed_labels = [seed_dataset.samples[i][1] for i in range(len(seed_dataset))]

            if K > 0:
                # build nearest neighbors
                self.net.eval()
                tgt_idxs = []
                tgt_embs = []
                tgt_labels = []
                tgt_data = []
                seq_query_loader = copy.deepcopy(seq_query_loader)
                seq_query_loader.dataset.transform = copy.deepcopy(self.tgt_loader.dataset.transform)
                with torch.no_grad():
                    for sample_ in seq_query_loader:
                        sample = copy.deepcopy(sample_)
                        del sample_
                        data, label, idx = sample[0], sample[1], sample[2]
                        data, label = data.to(self.device), label.to(self.device)
                        score, emb = self.net(data, with_emb=True)
                        tgt_embs.append(F.normalize(emb).detach().clone().cpu())
                        tgt_labels.append(label.cpu())
                        tgt_idxs.append(idx.cpu())
                        tgt_data.append(data.cpu())

                tgt_embs = torch.cat(tgt_embs)
                tgt_data = torch.cat(tgt_data)
                tgt_idxs = torch.cat(tgt_idxs)

        self.net.train()

        src_iter = iter(self.src_loader)
        iter_per_epoch = len(self.src_loader)

        for batch_idx in range(iter_per_epoch):
            if batch_idx % len(self.src_loader) == 0:
                src_iter = iter(self.src_loader)

            data_s, label_s, _ = next(src_iter)
            data_s, label_s = data_s.to(self.device), label_s.to(self.device)

            self.tgt_opt.zero_grad()
            output_s = self.net(data_s)
            loss = nn.CrossEntropyLoss()(output_s, label_s)

            if len(self.tgt_sup_loader) > 0:
                try:
                    data_ts, label_ts, idx_ts, *data_rand_ts = next(seed_iter)
                except:
                    seed_iter = iter(seed_loader)
                    data_ts, label_ts, idx_ts, *data_rand_ts = next(seed_iter)


                if len(data_rand_ts)>0:
                    for i, r_data in enumerate(data_rand_ts):
                        alpha = 0.2
                        mask = torch.FloatTensor(np.random.beta(alpha, alpha, size=(data_ts.shape[0], 1, 1, 1)))
                        data_ts = (data_ts * mask) + (r_data * (1 - mask))
                        data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
                        output_ts, emb_ts = self.net(data_ts, with_emb=True)
                        loss += nn.CrossEntropyLoss()(output_ts, label_ts)
                else:
                    data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
                    output_ts, emb_ts = self.net(data_ts, with_emb=True)
                    loss += nn.CrossEntropyLoss()(output_ts, label_ts)

            loss.backward()
            self.tgt_opt.step()

            if len(self.tgt_sup_loader) > 0 and K > 0 and len(seed_idxs) < tgt_embs.shape[0]:
                nn_idxs = torch.randint(0, tgt_data.shape[0], (data_ts.shape[0],)).to(self.device)

                data_nn = tgt_data[nn_idxs].to(self.device)

                with torch.no_grad():
                    output_nn, emb_nn = self.net(data_nn, with_emb=True)
                    prob_nn = torch.softmax(output_nn, dim=-1)
                    tgt_embs[nn_idxs] = F.normalize(emb_nn).detach().clone().cpu()

                conf_samples = []
                conf_idx = []
                conf_pl = []
                dist = np.eye(prob_nn.shape[-1])[np.array(seed_labels)].sum(0) + 1
                sp = 1 - dist / dist.max() + dist.min() / dist.max()

                for i in range(prob_nn.shape[0]):
                    idx = tgt_idxs[nn_idxs[i]].item()
                    pl_i = prob_nn[i].argmax(-1).item()
                    if np.random.random() <= sp[pl_i] and prob_nn[i].max(-1)[0] >= th and idx not in seed_idxs:
                        conf_samples.append((self.tgt_loader.dataset.samples[idx][0], pl_i))
                        conf_idx.append(idx)
                        conf_pl.append(pl_i)

                seed_dataset.add_item(conf_samples)
                seed_idxs.extend(conf_idx)
                seed_labels.extend(conf_pl)


@register_solver('LAA')
class LAASolver(BaseSolver):
    """
      Local context-aware Anchor set Augmentation (LAA)
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt,
                 ada_stage, device, cfg, **kwargs):
        super(LAASolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader,
                                         joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs)

    def solve(self, epoch, seq_query_loader):
        K = self.cfg.LADA.A_K
        th = self.cfg.LADA.A_TH

        # create an anchor set
        if len(self.tgt_sup_loader) > 0:
            tgt_sup_dataset = self.tgt_sup_loader.dataset
            tgt_sup_samples = [tgt_sup_dataset.samples[i] for i in self.tgt_sup_loader.sampler.indices]
            seed_dataset = ImageList(tgt_sup_samples, root=tgt_sup_dataset.root, transform=tgt_sup_dataset.transform)
            seed_dataset.rand_transform = rand_transform
            seed_dataset.rand_num = self.cfg.LADA.A_RAND_NUM
            seed_loader = torch.utils.data.DataLoader(seed_dataset, shuffle=True,
                                          batch_size=self.tgt_sup_loader.batch_size, num_workers=self.tgt_sup_loader.num_workers)
            seed_idxs = self.tgt_sup_loader.sampler.indices.tolist()
            seed_iter = iter(seed_loader)
            seed_labels = [seed_dataset.samples[i][1] for i in range(len(seed_dataset))]

            if K > 0:
                # build nearest neighbors
                self.net.eval()
                tgt_idxs = []
                tgt_embs = []
                tgt_labels = []
                tgt_data = []
                seq_query_loader = copy.deepcopy(seq_query_loader)
                seq_query_loader.dataset.transform = copy.deepcopy(self.tgt_loader.dataset.transform)
                with torch.no_grad():
                    for sample_ in seq_query_loader:
                        sample = copy.deepcopy(sample_)
                        del sample_
                        data, label, idx = sample[0], sample[1], sample[2]
                        data, label = data.to(self.device), label.to(self.device)
                        score, emb = self.net(data, with_emb=True)
                        tgt_embs.append(F.normalize(emb).detach().clone().cpu())
                        tgt_labels.append(label.cpu())
                        tgt_idxs.append(idx.cpu())
                        tgt_data.append(data.cpu())

                tgt_embs = torch.cat(tgt_embs)
                tgt_data = torch.cat(tgt_data)
                tgt_idxs = torch.cat(tgt_idxs)

        self.net.train()

        src_iter = iter(self.src_loader)
        iter_per_epoch = len(self.src_loader)

        for batch_idx in range(iter_per_epoch):
            if batch_idx % len(self.src_loader) == 0:
                src_iter = iter(self.src_loader)

            data_s, label_s, _ = next(src_iter)
            data_s, label_s = data_s.to(self.device), label_s.to(self.device)

            self.tgt_opt.zero_grad()
            output_s = self.net(data_s)
            loss = nn.CrossEntropyLoss()(output_s, label_s)

            if len(self.tgt_sup_loader) > 0:
                try:
                    data_ts, label_ts, idx_ts, *data_rand_ts = next(seed_iter)
                except:
                    seed_iter = iter(seed_loader)
                    data_ts, label_ts, idx_ts, *data_rand_ts = next(seed_iter)

                if len(data_rand_ts) > 0:
                    for i, r_data in enumerate(data_rand_ts):
                        alpha = 0.2
                        mask = torch.FloatTensor(np.random.beta(alpha, alpha, size=(data_ts.shape[0], 1, 1, 1)))
                        data_ts = (data_ts * mask) + (r_data * (1 - mask))
                        data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
                        output_ts, emb_ts = self.net(data_ts, with_emb=True)
                        loss += nn.CrossEntropyLoss()(output_ts, label_ts)
                else:
                    data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
                    output_ts, emb_ts = self.net(data_ts, with_emb=True)
                    loss += nn.CrossEntropyLoss()(output_ts, label_ts)

            loss.backward()
            self.tgt_opt.step()

            if len(self.tgt_sup_loader) > 0 and K > 0 and len(seed_idxs) < tgt_embs.shape[0]:
                mask = torch.ones(tgt_embs.shape[0])
                re_idxs = tgt_idxs[mask == 1]

                sim = F.normalize(emb_ts.cpu()).mm(tgt_embs[re_idxs].transpose(1, 0))
                sim_topk, topk = torch.topk(sim, k=K, dim=1)

                rand_nn = torch.randint(0, topk.shape[1], (topk.shape[0], 1))
                nn_idxs = torch.gather(topk, dim=-1, index=rand_nn).squeeze(1)
                nn_idxs = re_idxs[nn_idxs]

                data_nn = tgt_data[nn_idxs].to(self.device)

                with torch.no_grad():
                    output_nn, emb_nn = self.net(data_nn, with_emb=True)
                    prob_nn = torch.softmax(output_nn, dim=-1)
                    tgt_embs[nn_idxs] = F.normalize(emb_nn).detach().clone().cpu()

                conf_samples = []
                conf_idx = []
                conf_pl = []
                dist = np.eye(prob_nn.shape[-1])[np.array(seed_labels)].sum(0) + 1
                dist = dist / dist.max()
                sp = 1 - dist / dist.max() + dist.min() / dist.max()

                for i in range(prob_nn.shape[0]):
                    idx = tgt_idxs[nn_idxs[i]].item()
                    pl_i = prob_nn[i].argmax(-1).item()
                    if np.random.random() <= sp[pl_i] and prob_nn[i].max(-1)[0] >= th and idx not in seed_idxs:
                        conf_samples.append((self.tgt_loader.dataset.samples[idx][0], pl_i))
                        conf_idx.append(idx)
                        conf_pl.append(pl_i)

                seed_dataset.add_item(conf_samples)
                seed_idxs.extend(conf_idx)
                seed_labels.extend(conf_pl)

