import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.image_list import ImageList
from .solver import BaseSolver, register_solver
from collections import defaultdict
import numpy as np
from dataset.transform import rand_transform2

def get_losses_unlabeled(net, im_data, im_data_bar, im_data_bar2, target, BCE, w_cons, device):
    """ Get losses for unlabeled samples."""

    output, feat = net(im_data, with_emb=True, reverse_grad=True)
    output_bar, feat_bar = net(im_data_bar, with_emb=True, reverse_grad=True)
    prob, prob_bar = F.softmax(output, dim=1), F.softmax(output_bar, dim=1)

    # loss for adversarial adpative clustering
    aac_loss = advbce_unlabeled(target=target, feat=feat, prob=prob, prob_bar=prob_bar, device=device, bce=BCE)

    output = net.forward_emb(feat)
    output_bar = net.forward_emb(feat_bar)
    output_bar2 = net(im_data_bar2)

    prob = F.softmax(output, dim=1)
    prob_bar = F.softmax(output_bar, dim=1)
    prob_bar2 = F.softmax(output_bar2, dim=1)

    max_probs, pseudo_labels = torch.max(prob.detach_(), dim=-1)
    mask = max_probs.ge(0.95).float()

    # loss for pseudo labeling
    pl_loss = (F.cross_entropy(output_bar2, pseudo_labels, reduction='none') * mask).mean()

    # loss for consistency
    con_loss = w_cons * F.mse_loss(prob_bar, prob_bar2)

    return aac_loss, pl_loss, con_loss


def advbce_unlabeled(target, feat, prob, prob_bar, device, bce):
    """ Construct adversarial adpative clustering loss."""
    target_ulb = pairwise_target(feat, target, device)
    prob_bottleneck_row, _ = PairEnum2D(prob)
    _, prob_bottleneck_col = PairEnum2D(prob_bar)
    adv_bce_loss = -bce(prob_bottleneck_row, prob_bottleneck_col, target_ulb)
    return adv_bce_loss


def pairwise_target(feat, target, device, topk=5):
    """ Produce pairwise similarity label."""
    feat_detach = feat.detach()
    # For unlabeled data
    if target is None:
        rank_feat = feat_detach
        rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
        rank_idx1, rank_idx2 = PairEnum2D(rank_idx)
        rank_idx1, rank_idx2 = rank_idx1[:, :topk], rank_idx2[:, :topk]
        rank_idx1, _ = torch.sort(rank_idx1, dim=1)
        rank_idx2, _ = torch.sort(rank_idx2, dim=1)
        rank_diff = rank_idx1 - rank_idx2
        rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
        target_ulb = torch.ones_like(rank_diff).float().to(device)
        target_ulb[rank_diff > 0] = 0
    # For labeled data
    elif target is not None:
        target_row, target_col = PairEnum1D(target)
        target_ulb = torch.zeros(target.size(0) * target.size(0)).float().to(device)
        target_ulb[target_row == target_col] = 1
    else:
        raise ValueError('Please check your target.')
    return target_ulb


def PairEnum1D(x):
    """ Enumerate all pairs of feature in x with 1 dimension."""
    assert x.ndimension() == 1, 'Input dimension must be 1'
    x1 = x.repeat(x.size(0), )
    x2 = x.repeat(x.size(0)).view(-1, x.size(0)).transpose(1, 0).reshape(-1)
    return x1, x2


def PairEnum2D(x):
    """ Enumerate all pairs of feature in x with 2 dimensions."""
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    return x1, x2


def sigmoid_rampup(current, rampup_length):
    """ Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class BCE(nn.Module):
    eps = 1e-7

    def forward(self, prob1, prob2, simi):
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


class BCE_softlabels(nn.Module):
    """ Construct binary cross-entropy loss."""
    eps = 1e-7

    def forward(self, prob1, prob2, simi):
        P = prob1.mul_(prob2)
        P = P.sum(1)
        neglogP = - (simi * torch.log(P + BCE.eps) + (1. - simi) * torch.log(1. - P + BCE.eps))
        return neglogP.mean()


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=0.0001,
                     power=0.75, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (- power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(- alpha * iter_num / max_iter)) -
                    (high - low) + low)


@register_solver('CDAC')
class CDACSolver(BaseSolver):
    """
    Implements Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation: https://arxiv.org/abs/2104.09415
    https://github.com/lijichang/CVPR2021-SSDA
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt,
                 ada_stage, device, cfg, **kwargs):
        super(CDACSolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader,
                                            joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs)

    def solve(self, epoch):
        self.net.train()

        self.tgt_unsup_loader.dataset.rand_transform = rand_transform2
        self.tgt_unsup_loader.dataset.rand_num = 2

        data_iter_s = iter(self.src_loader)
        data_iter_t = iter(self.tgt_sup_loader)
        data_iter_t_unl = iter(self.tgt_unsup_loader)

        len_train_source = len(self.src_loader)
        len_train_target = len(self.tgt_sup_loader)
        len_train_target_semi = len(self.tgt_unsup_loader)

        BCE = BCE_softlabels().to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)

        iter_per_epoch = len(self.src_loader)
        for batch_idx in range(iter_per_epoch):
            rampup = sigmoid_rampup(batch_idx+epoch*iter_per_epoch, 20000)
            w_cons = 30.0 * rampup

            self.tgt_opt = inv_lr_scheduler([0.1, 1.0, 1.0], self.tgt_opt, batch_idx+epoch*iter_per_epoch,
                                            init_lr=0.01)

            if len(self.tgt_sup_loader) > 0:
                if batch_idx % len_train_target == 0:
                    data_iter_t = iter(self.tgt_sup_loader)
                if batch_idx % len_train_target_semi == 0:
                    data_iter_t_unl = iter(self.tgt_unsup_loader)
                if batch_idx % len_train_source == 0:
                    data_iter_s = iter(self.src_loader)
                data_t = next(data_iter_t)
                data_t_unl = next(data_iter_t_unl)
                data_s = next(data_iter_s)

                # load labeled source data
                x_s, target_s = data_s[0], data_s[1]
                im_data_s = x_s.to(self.device)
                gt_labels_s = target_s.to(self.device)

                # load labeled target data
                x_t, target_t = data_t[0], data_t[1]
                im_data_t = x_t.to(self.device)
                gt_labels_t = target_t.to(self.device)

                # load unlabeled target data
                x_tu, x_bar_tu, x_bar2_tu = data_t_unl[0], data_t_unl[3], data_t_unl[4]
                im_data_tu = x_tu.to(self.device)
                im_data_bar_tu = x_bar_tu.to(self.device)
                im_data_bar2_tu = x_bar2_tu.to(self.device)

                self.tgt_opt.zero_grad()
                # construct losses for overall labeled data
                data = torch.cat((im_data_s, im_data_t), 0)
                target = torch.cat((gt_labels_s, gt_labels_t), 0)
                out1 = self.net(data)
                ce_loss = criterion(out1, target)

                ce_loss.backward(retain_graph=True)
                self.tgt_opt.step()
                self.tgt_opt.zero_grad()

                # construct losses for unlabeled target data
                aac_loss, pl_loss, con_loss = get_losses_unlabeled(self.net, im_data=im_data_tu, im_data_bar=im_data_bar_tu,
                                                                   im_data_bar2=im_data_bar2_tu, target=None, BCE=BCE,
                                                                   w_cons=w_cons, device=self.device)
                loss = (aac_loss + pl_loss + con_loss) * self.cfg.ADA.UNSUP_WT * 10
            else:
                if batch_idx % len_train_source == 0:
                    data_iter_s = iter(self.src_loader)
                data_s, label_s, _ = next(data_iter_s)
                data_s, label_s = data_s.to(self.device), label_s.to(self.device)

                self.tgt_opt.zero_grad()
                output_s = self.net(data_s)
                loss = nn.CrossEntropyLoss()(output_s, label_s) * self.cfg.ADA.SRC_SUP_WT

            loss.backward()
            self.tgt_opt.step()




