import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
import numpy as np
from .solver import BaseSolver, register_solver

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


@register_solver('MCC')
class MCCSolver(BaseSolver):
    """
    Implements MCC from Minimum Class Confusion for Versatile Domain Adaptation: https://arxiv.org/abs/1912.03699
    https://github.com/thuml/Versatile-Domain-Adaptation
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt,
                 ada_stage, device, cfg, **kwargs):
        super(MCCSolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader,
                                         joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs)

    def solve(self, epoch):
        src_iter = iter(self.src_loader)
        tgt_un_iter = iter(self.tgt_unsup_loader)
        tgt_s_iter = iter(self.tgt_sup_loader)
        iter_per_epoch = len(self.src_loader)

        self.net.train()

        for batch_idx in range(iter_per_epoch):
            if batch_idx % len(self.src_loader) == 0:
                src_iter = iter(self.src_loader)

            if batch_idx % len(self.tgt_unsup_loader) == 0:
                tgt_un_iter = iter(self.tgt_unsup_loader)

            data_s, label_s, _ = next(src_iter)
            data_s, label_s = data_s.to(self.device), label_s.to(self.device)

            self.tgt_opt.zero_grad()
            output_s = self.net(data_s)
            loss = nn.CrossEntropyLoss()(output_s, label_s) * self.cfg.ADA.SRC_SUP_WT

            if len(self.tgt_sup_loader) > 0:
                try:
                    data_ts, label_ts, idx_ts = next(tgt_s_iter)
                except:
                    tgt_s_iter = iter(self.tgt_sup_loader)
                    data_ts, label_ts, idx_ts = next(tgt_s_iter)

                data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
                output_ts = self.net(data_ts)

                loss += nn.CrossEntropyLoss()(output_ts, label_ts)

            data_tu, label_tu, _ = next(tgt_un_iter)
            data_tu, label_tu = data_tu.to(self.device), label_tu.to(self.device)
            output_tu = self.net(data_tu)

            outputs_target_temp = output_tu / self.cfg.MODEL.TEMP
            target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
            target_entropy_weight = Entropy(target_softmax_out_temp).detach()
            target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
            target_entropy_weight = self.cfg.DATALOADER.BATCH_SIZE * target_entropy_weight / torch.sum(target_entropy_weight)
            cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
                target_softmax_out_temp)
            cov_matrix_t = cov_matrix_t / (torch.sum(cov_matrix_t, dim=1)+1e-12)
            mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / self.cfg.DATASET.NUM_CLASS

            loss += mcc_loss

            loss.backward()
            self.tgt_opt.step()



