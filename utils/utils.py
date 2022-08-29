import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
# from model import get_model
import torch.optim as optim
# from solver import get_solver
import logging

def resetRNGseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, device, train_loader, optimizer, epoch):
    """
    Test model on provided data for single epoch
    """
    model.train()
    total_loss, correct = 0.0, 0
    for batch_idx, (data, target, _) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        corr = pred.eq(target.view_as(pred)).sum().item()
        correct += corr
        loss.backward()
        optimizer.step()

    train_acc = 100. * correct / len(train_loader.sampler)
    avg_loss = total_loss / len(train_loader.sampler)
    logging.info('Train Epoch: {} | Avg. Loss: {:.3f} | Train Acc: {:.3f}'.format(epoch, avg_loss, train_acc))
    return avg_loss

def test(model, device, test_loader, split="target test"):
    """
    Test model on provided data
    """
    # logging.info('Evaluating model on {}...'.format(split))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            test_loss += loss.item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            corr = pred.eq(target.view_as(pred)).sum().item()
            correct += corr
            del loss, output

    test_loss /= len(test_loader.sampler)
    test_acc = 100. * correct / len(test_loader.sampler)

    return test_acc, test_loss


# def run_unsupervised_da(model, src_train_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, num_classes, device,
#                         cfg):
#     """
#     Unsupervised adaptation of source model to target at round 0
#     Returns:
#         Model post adaptation
#     """
#     source = cfg.DATASET.SOURCE_DOMAIN
#     target = cfg.DATASET.TARGET_DOMAIN
#     da_strat = cfg.ADA.DA
#
#     adapt_dir = os.path.join('checkpoints', 'adapt')
#     adapt_net_file = os.path.join(adapt_dir, '{}_{}_{}_{}.pth'.format(da_strat, source, target, cfg.MODEL.BACKBONE.NAME))
#
#     if not os.path.exists(adapt_dir):
#         os.makedirs(adapt_dir)
#
#     if os.path.exists(adapt_net_file):
#         logging.info('Found pretrained checkpoint, loading...')
#         adapt_model = get_model('AdaptNet', num_cls=num_classes, weights_init=adapt_net_file, model=cfg.MODEL.BACKBONE.NAME)
#     else:
#         logging.info('No pretrained checkpoint found, training...')
#         source_file = '{}_{}_source.pth'.format(source, cfg.MODEL.BACKBONE.NAME)
#         source_path = os.path.join('checkpoints', 'source', source_file)
#         adapt_model = get_model('AdaptNet', num_cls=num_classes, src_weights_init=source_path, model=cfg.MODEL.BACKBONE.NAME)
#         opt_net_tgt = optim.Adadelta(adapt_model.tgt_net.parameters(cfg.OPTIM.UDA_LR, cfg.OPTIM.BASE_LR_MULT), lr=cfg.OPTIM.UDA_LR, weight_decay=0.00001)
#         uda_solver = get_solver(da_strat, adapt_model.tgt_net, src_train_loader, tgt_sup_loader, tgt_unsup_loader,
#                                 train_idx, opt_net_tgt, 0, device, cfg)
#         for epoch in range(cfg.TRAINER.MAX_UDA_EPOCHS):
#             if da_strat == 'dann':
#                 opt_dis_adapt = optim.Adadelta(adapt_model.discriminator.parameters(), lr=cfg.OPTIM.UDA_LR, weight_decay=0.00001)
#                 uda_solver.solve(epoch, adapt_model.discriminator, opt_dis_adapt)
#             elif da_strat in ['mme', 'ft']:
#                 uda_solver.solve(epoch)
#         adapt_model.save(adapt_net_file)
#
#     model, src_model, discriminator = adapt_model.tgt_net, adapt_model.src_net, adapt_model.discriminator
#     return model, src_model, discriminator


def get_optim(name, *args, **kwargs):
    if name == 'Adadelta':
        return optim.Adadelta(*args, **kwargs)
    elif name == 'Adam':
        return optim.Adam(*args, **kwargs)
    elif name == 'SGD':
        return optim.SGD(*args, **kwargs, momentum=0.9, nesterov=True)