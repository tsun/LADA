import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from dataset.image_list import ImageList
from dataset.transform import rand_transform
from .solver import BaseSolver, register_solver

@register_solver('LADA')
class LADASolver(BaseSolver):
    """
      Implement Local contex-aware model adaptation (LMA)
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt,
                 ada_stage, device, cfg, **kwargs):
        super(LADASolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader,
                                         joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs)

    def solve(self, epoch, seq_query_loader):
        K = self.cfg.LADA.A_K

        # create an anchor set
        if len(self.tgt_sup_loader) > 0:
            tgt_sup_dataset = self.tgt_sup_loader.dataset
            tgt_sup_samples = [tgt_sup_dataset.samples[i] for i in self.tgt_sup_loader.sampler.indices]
            seed_dataset = ImageList(tgt_sup_samples, root=tgt_sup_dataset.root, transform=tgt_sup_dataset.transform)
            seed_dataset.rand_transform = rand_transform
            seed_dataset.rand_num = self.cfg.LADA.A_RAND_NUM
            seed_loader = torch.utils.data.DataLoader(seed_dataset, shuffle=True,
                                          batch_size=self.tgt_sup_loader.batch_size, num_workers=self.tgt_sup_loader.num_workers)
            seed_idx = self.tgt_sup_loader.sampler.indices.tolist()
            seed_iter = iter(seed_loader)

            if K > 0:
                # build nearest neighbors
                self.net.eval()
                tgt_idxs = []
                tgt_embs = []
                tgt_probs = []
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
                        tgt_probs.append(torch.softmax(score.detach().clone().cpu(),dim=-1))
                        tgt_labels.append(label.cpu())
                        tgt_idxs.append(idx.cpu())
                        tgt_data.append(data.cpu())

                tgt_embs = torch.cat(tgt_embs)
                tgt_probs = torch.cat(tgt_probs)
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

                data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
                output_ts, emb_ts = self.net(data_ts, with_emb=True)
                prob_ts = torch.softmax(output_ts, dim=-1)
                loss += nn.CrossEntropyLoss()(output_ts, label_ts)

                for i, r_data in enumerate(data_rand_ts):
                    r_data = r_data.to(self.device)
                    output_rand_ts = self.net(r_data)
                    loss += nn.CrossEntropyLoss()(output_rand_ts, label_ts)

                if K>0:
                    sim = F.normalize(emb_ts.cpu()).mm(tgt_embs.transpose(1,0))
                    sim_topk, topk = torch.topk(sim, k=K+1, dim=1)
                    sim_topk, topk = sim_topk[:, 1:].to(self.device), topk[:, 1:].to(self.device)
                    wgt_topk = (sim_topk / sim_topk.sum(dim=1, keepdim=True)).detach().clone()
                    knn_loss = -((tgt_probs[topk].to(self.device) * prob_ts.unsqueeze(1)).sum(-1) * wgt_topk).sum(-1).mean()
                    loss += self.cfg.LADA.A_ALPHA * knn_loss

                    # add confident samples to anchor set
                    rand_nn = torch.randint(0, topk.shape[1], (topk.shape[0], 1)).to(self.device)
                    nn_idxs = torch.gather(topk, dim=-1, index=rand_nn).squeeze(1)
                    data_nn = tgt_data[nn_idxs].to(self.device)
                    output_nn, emb_nn = self.net(data_nn, with_emb=True)
                    prob_nn = torch.softmax(output_nn, dim=-1)
                    conf_samples = []
                    conf_idx = []
                    for i in range(prob_nn.shape[0]):
                        idx = tgt_idxs[nn_idxs[i]].item()
                        if prob_nn[i].max(-1)[0] > self.cfg.LADA.A_TH and idx not in seed_idx:
                            conf_samples.append((self.tgt_loader.dataset.samples[idx][0], prob_nn[i].argmax(-1).item()))
                            conf_idx.append(idx)

                    seed_dataset.add_item(conf_samples)
                    seed_idx.extend(conf_idx)


            loss.backward()
            self.tgt_opt.step()

