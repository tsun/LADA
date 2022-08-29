import torch
import torch.nn as nn

from .utils import ConditionalEntropyLoss
from model.grl import GradientReverseFunction


solvers = {}
def register_solver(name):
    def decorator(cls):
        solvers[name] = cls
        return cls
    return decorator

def get_solver(name, *args, kwargs={}):
    solver = solvers[name](*args, **kwargs)
    return solver

class BaseSolver:
    """
    Base DA solver class
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt, ada_stage, device, cfg):
        self.net = net
        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self.tgt_sup_loader = tgt_sup_loader
        self.tgt_unsup_loader = tgt_unsup_loader
        self.joint_sup_loader = joint_sup_loader
        self.tgt_opt = tgt_opt
        self.ada_stage = ada_stage
        self.device = device
        self.cfg = cfg

    def solve(self, epoch):
        pass

@register_solver('ft_joint')
class JointFTSolver(BaseSolver):
    """
    Finetune on target labels
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs):
        super(JointFTSolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt,
                                             ada_stage, device, cfg, **kwargs)

    def solve(self, epoch):
        """
        Finetune on source and target labels jointly
        """
        self.net.train()
        joint_sup_iter = iter(self.joint_sup_loader)

        while True:
            try:
                data, target, _ = next(joint_sup_iter)
                data, target = data.to(self.device), target.to(self.device)
            except:
                break

            self.tgt_opt.zero_grad()
            output = self.net(data)
            loss = nn.CrossEntropyLoss()(output, target)

            loss.backward()
            self.tgt_opt.step()


@register_solver('ft_tgt')
class TargetFTSolver(BaseSolver):
    """
    Finetune on target labels
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs):
        super(TargetFTSolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt,
                                             ada_stage, device, cfg, **kwargs)

    def solve(self, epoch):
        """
        Finetune on target labels
        """
        self.net.train()
        if self.ada_stage: tgt_sup_iter = iter(self.tgt_sup_loader)

        while True:
            try:
                data_t, target_t, _ = next(tgt_sup_iter)
                data_t, target_t = data_t.to(self.device), target_t.to(self.device)
            except:
                break

            self.tgt_opt.zero_grad()
            output = self.net(data_t)
            loss = nn.CrossEntropyLoss()(output, target_t)
            loss.backward()
            self.tgt_opt.step()


@register_solver('ft')
class FTSolver(BaseSolver):
    """
    Finetune on source and target labels with separate loaders
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs):
        super(FTSolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt, ada_stage,
                                        device, cfg, **kwargs)

    def solve(self, epoch):
        self.net.train()

        if self.ada_stage:
            src_sup_wt = 1.0
        else:
            src_sup_wt = self.cfg.ADA.SRC_SUP_WT

        tgt_sup_wt = self.cfg.ADA.TGT_SUP_WT

        tgt_sup_iter = iter(self.tgt_sup_loader)

        for batch_idx, (data_s, label_s, _) in enumerate(self.src_loader):
            data_s, label_s = data_s.to(self.device), label_s.to(self.device)

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
                xeloss_tgt = tgt_sup_wt * nn.CrossEntropyLoss()(score_ts, label_ts)

            xeloss = xeloss_src + xeloss_tgt
            xeloss.backward()
            self.tgt_opt.step()


@register_solver('dann')
class DANNSolver(BaseSolver):
    """
    Implements DANN from Unsupervised Domain Adaptation by Backpropagation: https://arxiv.org/abs/1409.7495
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs):
        super(DANNSolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt,
                                         ada_stage, device, cfg, **kwargs)

    def solve(self, epoch, disc, disc_opt):
        """
        Semisupervised adaptation via DANN: XE on labeled source + XE on labeled target + \
                                    ent. minimization on target + DANN on source<->target
        """
        gan_criterion = nn.CrossEntropyLoss()
        cent = ConditionalEntropyLoss().to(self.device)

        self.net.train()
        disc.train()

        if not self.ada_stage:
            src_sup_wt, lambda_unsup, lambda_cent = 1.0, 0.1, 0.01  # Hardcoded for unsupervised DA
        else:
            src_sup_wt, lambda_unsup, lambda_cent = self.cfg.ADA.SRC_SUP_WT, self.cfg.ADA.UNSUP_WT, self.cfg.ADA.CEN_WT
            tgt_sup_iter = iter(self.tgt_sup_loader)

        joint_loader = zip(self.src_loader, self.tgt_loader) # changed to tgt_loader to be consistent with CLUE implementation
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

            # zero gradients for optimizers
            self.tgt_opt.zero_grad()
            disc_opt.zero_grad()

            # Train with target labels
            score_s = self.net(data_s)
            xeloss_src = src_sup_wt * nn.CrossEntropyLoss()(score_s, label_s)

            xeloss_tgt = 0
            if self.ada_stage and data_ts is not None:
                score_ts = self.net(data_ts)
                xeloss_tgt = nn.CrossEntropyLoss()(score_ts, label_ts)

            # extract and concat features
            score_tu = self.net(data_tu)
            f = torch.cat((score_s, score_tu), 0)

            # predict with discriminator
            f_rev = GradientReverseFunction.apply(f)
            pred_concat = disc(f_rev)

            target_dom_s = torch.ones(len(data_s)).long().to(self.device)
            target_dom_t = torch.zeros(len(data_tu)).long().to(self.device)
            label_concat = torch.cat((target_dom_s, target_dom_t), 0)

            # compute loss for disciminator
            loss_domain = gan_criterion(pred_concat, label_concat)
            loss_cent = cent(score_tu)

            loss_final = (xeloss_src + xeloss_tgt) + (lambda_unsup * loss_domain) + (lambda_cent * loss_cent)

            loss_final.backward()

            self.tgt_opt.step()
            disc_opt.step()


