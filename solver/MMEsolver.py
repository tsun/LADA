import torch
import torch.nn as nn
import torch.nn.functional as F

from .solver import BaseSolver, register_solver

@register_solver('mme')
class MMESolver(BaseSolver):
    """
    Implements MME from Semi-supervised Domain Adaptation via Minimax Entropy: https://arxiv.org/abs/1904.06487
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs):
        super(MMESolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt, ada_stage,
                                        device, cfg, **kwargs)

    def solve(self, epoch):
        """
        Semisupervised adaptation via MME: XE on labeled source + XE on labeled target + \
                                        adversarial ent. minimization on unlabeled target
        """
        self.net.train()

        if not self.ada_stage:
            src_sup_wt, lambda_unsup = 1.0, 0.1
        else:
            src_sup_wt, lambda_unsup = self.cfg.ADA.SRC_SUP_WT, self.cfg.ADA.UNSUP_WT

        tgt_sup_iter = iter(self.tgt_sup_loader)

        joint_loader = zip(self.src_loader, self.tgt_unsup_loader) # changed to tgt_loader to be consistent with CLUE implementation
        for batch_idx, ((data_s, label_s, _), (data_tu, label_tu, _)) in enumerate(joint_loader):
            data_s, label_s = data_s.to(self.device), label_s.to(self.device)
            data_tu = data_tu.to(self.device)

            if self.ada_stage:
                try:
                    data_ts, label_ts, _ = next(tgt_sup_iter)
                    data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
                except:
                    # no labeled target data
                    try:
                        tgt_sup_iter = iter(self.tgt_sup_loader)
                        data_ts, label_ts, _ = next(tgt_sup_iter)
                        data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
                    except:
                        data_ts, label_ts = None, None

            # zero gradients for optimizer
            self.tgt_opt.zero_grad()

            # extract features
            score_s = self.net(data_s)
            xeloss_src = src_sup_wt * nn.CrossEntropyLoss()(score_s, label_s)

            xeloss_tgt = 0
            if self.ada_stage and data_ts is not None:
                score_ts = self.net(data_ts)
                xeloss_tgt = nn.CrossEntropyLoss()(score_ts, label_ts)

            xeloss = xeloss_src + xeloss_tgt

            xeloss.backward()
            self.tgt_opt.step()

            # Add adversarial entropy
            self.tgt_opt.zero_grad()

            score_tu = self.net(data_tu, reverse_grad=True)
            probs_tu = F.softmax(score_tu, dim=1)
            loss_adent = lambda_unsup * torch.mean(torch.sum(probs_tu * (torch.log(probs_tu + 1e-5)), 1))
            loss_adent.backward()

            self.tgt_opt.step()


