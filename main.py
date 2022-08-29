import argparse
import torch
import logging
import os
import time
import copy
import pprint as pp
from tqdm import tqdm, trange
from collections import defaultdict
import numpy as np
import shutil
import socket
hostName = socket.gethostname()
pid = os.getpid()

from config.defaults import _C as cfg
from utils.logger import init_logger
from utils.utils import resetRNGseed
from dataset.ASDADataset import ASDADataset
from dataset.transform import build_transforms

from model import get_model
import utils.utils as utils

from active.sampler import get_strategy
from active.budget import BudgetAllocator


# repeatability
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run_active_adaptation(cfg, source_model, src_dset, num_classes, device):
    source = cfg.DATASET.SOURCE_DOMAIN
    target = cfg.DATASET.TARGET_DOMAIN
    da_strat = cfg.ADA.DA
    al_strat = cfg.ADA.AL

    transforms = build_transforms(cfg, 'target')
    tgt_dset = ASDADataset(cfg.DATASET.NAME, cfg.DATASET.TARGET_DOMAIN, data_dir=cfg.DATASET.ROOT,
                           num_classes=cfg.DATASET.NUM_CLASS, batch_size=cfg.DATALOADER.BATCH_SIZE,
                           num_workers=cfg.DATALOADER.NUM_WORKERS, transforms=transforms)

    target_test_loader = tgt_dset.get_loaders()[2]

    # Evaluate source model on target test
    if cfg.TRAINER.EVAL_ACC:
        transfer_perf, _ = utils.test(source_model, device, target_test_loader)
        logging.info('{}->{} performance (Before {}): Task={:.2f}'.format(source, target, da_strat, transfer_perf))


    # Main Active DA loop
    logging.info('------------------------------------------------------')
    model_init = 'source' if cfg.TRAINER.TRAIN_ON_SOURCE else 'scratch'
    logging.info('Running strategy: Init={} AL={} DA={}'.format(model_init, al_strat, da_strat))
    logging.info('------------------------------------------------------')

    # Run unsupervised DA at round 0, where applicable
    start_perf = 0.

    # Instantiate active sampling strategy
    sampling_strategy = get_strategy(al_strat, src_dset, tgt_dset, source_model, device, num_classes, cfg)
    del source_model

    if cfg.TRAINER.MAX_UDA_EPOCHS > 0:
        target_model = sampling_strategy.train_uda(epochs=cfg.TRAINER.MAX_UDA_EPOCHS)

        # Evaluate adapted source model on target test
        if cfg.TRAINER.EVAL_ACC:
            start_perf, _ = utils.test(target_model, device, target_test_loader)
            logging.info('{}->{} performance (After {}): {:.2f}'.format(source, target, da_strat, start_perf))
            logging.info('------------------------------------------------------')


    # Run Active DA
    # Keep track of labeled vs unlabeled data
    idxs_lb = np.zeros(len(tgt_dset.train_idx), dtype=bool)

    budget = np.round(len(tgt_dset.train_idx) * cfg.ADA.BUDGET) if cfg.ADA.BUDGET <= 1.0 else np.round(cfg.ADA.BUDGET)
    budget_allocator = BudgetAllocator(budget=budget, cfg=cfg)

    tqdm_rat = trange(cfg.TRAINER.MAX_EPOCHS)
    target_accs = defaultdict(list)
    target_accs[0.0].append(start_perf)

    for epoch in tqdm_rat:

        curr_budget, used_budget = budget_allocator.get_budget(epoch)
        tqdm_rat.set_description('# Target labels={} Allowed labels={}'.format(used_budget, curr_budget))
        tqdm_rat.refresh()

        # Select instances via AL strategy
        if curr_budget > 0:
            logging.info('Selecting instances...')
            idxs = sampling_strategy.query(curr_budget, epoch)
            idxs_lb[idxs] = True
            sampling_strategy.update(idxs_lb)
        else:
            logging.info('No budget for current epoch, skipped...')

        # Update model with new data via DA strategy
        target_model = sampling_strategy.train(epoch=epoch)

        # Evaluate on target test and train splits
        if cfg.TRAINER.EVAL_ACC:
            test_perf, _ = utils.test(target_model, device, target_test_loader)
            out_str = '{}->{} Test performance (Epoch {}, # Target labels={:d}): {:.2f}'.format(source, target, epoch,
                                                                                int(curr_budget+used_budget), test_perf)
            logging.info(out_str)
            logging.info('------------------------------------------------------')

            target_accs[curr_budget+used_budget].append(test_perf)


    logging.info("\n{}".format(target_accs))

    return target_accs


def ADAtrain(cfg, task):
    logging.info("Running task: {}".format(task))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transforms = build_transforms(cfg, 'source')
    src_dset = ASDADataset(cfg.DATASET.NAME, cfg.DATASET.SOURCE_DOMAIN, data_dir=cfg.DATASET.ROOT,
                           num_classes=cfg.DATASET.NUM_CLASS, batch_size=cfg.DATALOADER.BATCH_SIZE,
                           num_workers=cfg.DATALOADER.NUM_WORKERS, transforms=transforms)

    src_train_loader, src_valid_loader, src_test_loader = src_dset.get_loaders(valid_type=cfg.DATASET.SOURCE_VALID_TYPE,
                           valid_ratio=cfg.DATASET.SOURCE_VALID_RATIO)

    # model
    source_model = get_model(cfg.MODEL.BACKBONE.NAME, num_cls=cfg.DATASET.NUM_CLASS, normalize=cfg.MODEL.NORMALIZE, temp=cfg.MODEL.TEMP).to(device)
    source_file = '{}_{}_source_{}.pth'.format(cfg.DATASET.SOURCE_DOMAIN, cfg.MODEL.BACKBONE.NAME, cfg.TRAINER.MAX_SOURCE_EPOCHS)
    source_dir = os.path.join('checkpoints', 'source')
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    source_path = os.path.join(source_dir, source_file)
    best_source_file = '{}_{}_source_best_{}.pth'.format(cfg.DATASET.SOURCE_DOMAIN, cfg.MODEL.BACKBONE.NAME, cfg.TRAINER.MAX_SOURCE_EPOCHS)
    best_source_path = os.path.join(source_dir, best_source_file)

    if cfg.TRAINER.TRAIN_ON_SOURCE and cfg.TRAINER.MAX_SOURCE_EPOCHS>0:
        if cfg.TRAINER.LOAD_FROM_CHECKPOINT and os.path.exists(source_path):
            logging.info('Loading source checkpoint: {}'.format(source_path))
            source_model.load_state_dict(torch.load(source_path, map_location=device), strict=False)
            best_source_model = source_model
        else:
            logging.info('Training {} model...'.format(cfg.DATASET.SOURCE_DOMAIN))
            best_val_acc, best_source_model = 0.0, None
            source_optimizer = utils.get_optim(cfg.OPTIM.SOURCE_NAME, source_model.parameters(cfg.OPTIM.SOURCE_LR, cfg.OPTIM.BASE_LR_MULT), lr=cfg.OPTIM.SOURCE_LR)

            for epoch in range(cfg.TRAINER.MAX_SOURCE_EPOCHS):
                utils.train(source_model, device, src_train_loader, source_optimizer, epoch)

                val_acc, _ = utils.test(source_model, device, src_valid_loader, split="source valid")
                logging.info('[Epoch: {}] Valid Accuracy: {:.3f} '.format(epoch, val_acc))

                if (val_acc > best_val_acc):
                    best_val_acc = val_acc
                    best_source_model = copy.deepcopy(source_model)
                    torch.save(best_source_model.state_dict(), best_source_path)

            del source_model
            # rename file in case of abnormal exit
            shutil.move(best_source_path, source_path)
    else:
        best_source_model = source_model

    # Evaluate on source test set
    if cfg.TRAINER.EVAL_ACC:
        test_acc, _ = utils.test(best_source_model, device, src_test_loader, split="source test")
        logging.info('{} Test Accuracy: {:.3f} '.format(cfg.DATASET.SOURCE_DOMAIN, test_acc))

    # Run active adaptation experiments
    target_accs = run_active_adaptation(cfg, best_source_model, src_dset, cfg.DATASET.NUM_CLASS, device)
    pp.pprint(target_accs)
    


def main():
    parser = argparse.ArgumentParser(description='Optimal Budget Allocation for Active Domain Adaptation')
    parser.add_argument('--cfg', default='', metavar='FILE', help='path to config file', type=str)
    parser.add_argument('--timestamp', default=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
                        type=str, help='timestamp')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
    parser.add_argument('--note', default=None, type=str, help='note to experiment')
    parser.add_argument('--log', default='./log', type=str, help='logging directory')
    parser.add_argument('--nolog', action='store_true', help='whether use logger')
    parser.add_argument("opts", help="Modify config options using the command-line",
                         default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger = "{}_{}_{}_{}".format(args.timestamp, cfg.DATASET.NAME, cfg.ADA.AL, cfg.ADA.DA) if not args.nolog else None
    init_logger(logger, dir=args.log)

    if args.note is not None:
        logging.info("Experiment note : {}".format(args.note))
        logging.info('------------------------------------------------------')
    logging.info("Running on {} gpu={} pid={}".format(hostName, args.gpu, pid))
    logging.info(cfg)
    logging.info('------------------------------------------------------')

    if type(cfg.SEED) is tuple or type(cfg.SEED) is list:
        seeds = cfg.SEED
    else:
        seeds = [cfg.SEED]

    for seed in seeds:
        logging.info("Using random seed: {}".format(seed))
        resetRNGseed(seed)

        if cfg.ADA.TASKS is not None:
            ada_tasks = cfg.ADA.TASKS
        else:
            ada_tasks = [[source, target] for source in cfg.DATASET.SOURCE_DOMAINS
                         for target in cfg.DATASET.TARGET_DOMAINS if source != target]

        for [source, target] in ada_tasks:
            cfg.DATASET.SOURCE_DOMAIN = source
            cfg.DATASET.TARGET_DOMAIN = target

            cfg.freeze()
            ADAtrain(cfg, task=source + '-->' + target)
            cfg.defrost()


if __name__ == '__main__':
    main()