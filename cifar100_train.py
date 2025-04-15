from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# new add
from dataset import DatasetMaker
from conformal import tps, aps, new_aps, raps
import torch.nn.functional as F
from focal_loss import FocalLoss, FocalLossAdaptive, EntropyLoss
from wsc import wsc_unbiased
from sscv import get_violation
import math
import copy
import multiprocessing

import warnings
warnings.filterwarnings('ignore')


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=128, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=1024, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
# parser.add_argument('--calib-batch', default=512, type=int, metavar='N',
#                     help='calib batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, default=2024, help='manual seed')
parser.add_argument('-e', '--evaluate', type=bool, default=False,
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# new add
parser.add_argument('--correction', default='cor', type=str)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# new add
cnt = 5

alpha = 0.1
size_loss_weight = 0.1
tau = 0.001
target_size = 1.0

CP_score = 'aps'
if args.correction == 'conf':
    loss_score = 'optimal_loss'
else:
    loss_score = 'apsloss' # 'tpsloss' 'focal_tpsloss_2.0' 'focal_tpsloss_adaptive' 'uncertain_loss' 'optimal_loss'

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)




def run_calculate_entropy(dataset, T=1.0):
    def calculate_entropy(probabilities):
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log(p, 2)
        return entropy

    entropy_list = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = args.test_batch, shuffle=False, num_workers=args.workers)
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        smx = torch.nn.Softmax(dim = 1)(inputs/T).detach().cpu().numpy()
        for i in range(smx.shape[0]):
            entropy_list.append(calculate_entropy(smx[i]))

    return np.mean(entropy_list)

def platt_logits(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.4]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        # print(iter, T.item())
        if abs(T_old - T.item()) < epsilon:
            break
    return T

def compute_conform(index, dataset_logits, temp, is_platt, cmodel, T):
    torch.manual_seed(index)
    logits_calib, logits_test = data.random_split(dataset_logits, temp)
    # Prepare the loaders
    logits_calib_loader = torch.utils.data.DataLoader(logits_calib, batch_size = args.train_batch, shuffle=True, num_workers=0)
    logits_test_loader = torch.utils.data.DataLoader(logits_test, batch_size = args.test_batch, shuffle=False, num_workers=0)
    T = platt_logits(cmodel, logits_calib_loader).cpu() if is_platt else T
    
    for batch_idx, (inputs, targets) in enumerate(logits_calib_loader):
        if batch_idx == 0:
            base_model_smx_calib = torch.nn.Softmax(dim = 1)(inputs/T).detach().cpu().numpy()
            labels_calib = targets.detach().cpu().numpy()
        else:
            base_model_smx_calib = np.concatenate((base_model_smx_calib, torch.nn.Softmax(dim = 1)(inputs/T).detach().cpu().numpy()), axis=0)
            labels_calib = np.concatenate((labels_calib, targets.detach().cpu().numpy()), axis=0)

    for batch_idx, (inputs, targets) in enumerate(logits_test_loader):
        if batch_idx == 0:
            base_model_smx_test = torch.nn.Softmax(dim = 1)(inputs/T).detach().cpu().numpy()
            labels_test = targets.detach().cpu().numpy()
        else:
            base_model_smx_test = np.concatenate((base_model_smx_test, torch.nn.Softmax(dim = 1)(inputs/T).detach().cpu().numpy()), axis=0)
            labels_test = np.concatenate((labels_test, targets.detach().cpu().numpy()), axis=0)
    
    if CP_score == 'tps':
        cov, eff = tps(base_model_smx_calib, base_model_smx_test, labels_calib, labels_test, len(logits_calib), alpha)
    elif CP_score == 'aps':
        _, cov, eff, pos, neg, pos_eff, neg_eff, qhat = aps(base_model_smx_calib, base_model_smx_test, labels_calib, labels_test, len(logits_calib), alpha)
    elif CP_score == 'new_aps':
        _, cov, eff, pos, neg, pos_eff, neg_eff, qhat = new_aps(base_model_smx_calib, base_model_smx_test, labels_calib, labels_test, len(logits_calib), alpha)
    elif CP_score == 'raps':
        cov, eff = raps(base_model_smx_calib, base_model_smx_test, labels_calib, labels_test, len(logits_calib), alpha)
    else:
        print('No such CP score!')
    
        # num_classes = prediction_sets.shape[1]  # 类别数

    if is_platt:
        return cov, eff, logits_calib, logits_test, T
    else:
        return cov, eff, pos, neg, pos_eff, neg_eff, qhat
    
def run_compute_conform(params):
    return compute_conform(params[0], params[1], params[2], params[3], params[4], params[5])

def conformal_prediction(dataset_logits, temp, is_platt=False, cmodel=None, T=1.0, N=10):
    covs = []
    effs = []
    class_coverage_list = []

    n_iters = N

    if is_platt:
        for i in range(n_iters):
            cov, eff, logits_calib, logits_test, T = compute_conform(i, dataset_logits, temp, is_platt, cmodel, T=None)
            
            covs.append(cov)
            effs.append(eff)
    else:
        params = []
        context = torch.multiprocessing.get_context("spawn")
        for i in range(n_iters):
            params.append((i, dataset_logits, temp, is_platt, cmodel, T))

        pool = context.Pool(processes=10)
        results = pool.map(run_compute_conform, params)
        pool.close()
        pool.join()
        
        covs = np.array(results)[:, 0]
        effs = np.array(results)[:, 1]
        pos = np.array(results)[:, 2]
        neg = np.array(results)[:, 3]
        pos_eff = np.array(results)[:, 4]
        neg_eff = np.array(results)[:, 5]
        qhat = np.array(results)[:, 6]

        print(effs)

    print('var of cov: ', np.var(covs), 'var of eff: ', np.var(effs))
    if is_platt:
        return np.mean(covs), np.mean(effs), logits_calib, logits_test, T
    else:
        return np.mean(covs), np.mean(effs), np.mean(pos), np.mean(neg), np.mean(pos_eff), np.mean(neg_eff), np.mean(qhat)


def main():
    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_valid_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    dataset1 = dataloader(root='./data', train=True, download=True, transform=None)
    dataset2 = dataloader(root='./data', train=False, download=False, transform=None)
    dataset = data.ConcatDataset([dataset1, dataset2])

    final_dict_d = {}
    final_dict_d['cov'] = []
    final_dict_d['eff'] = []
    final_dict_d['entropy'] = []
    final_dict_d['T'] = []

    final_dict = {}
    final_dict['cov'] = []
    final_dict['eff'] = []
    final_dict['entropy'] = []


    for run in range(0, cnt):
        # reset
        best_acc = 0
        best_eff = 10000
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        state['lr'] = args.lr

        if not os.path.isdir(args.checkpoint + f'-{run}') and not args.resume:
            mkdir_p(args.checkpoint + f'-{run}')

        if args.resume and not os.path.isdir(args.resume + f'-{args.correction}-{run}'):
            mkdir_p(args.resume + f'-{args.correction}-{run}')
        
        # Split testset
        torch.manual_seed(run)
        temp = tuple([int(0.6*len(dataset)), int(0.1*len(dataset)), len(dataset) - int(0.6*len(dataset)) - int(0.1*len(dataset))])
        trainset, validset, testset = data.random_split(dataset, temp)

        trainset = DatasetMaker(trainset, transform_train)
        validset = DatasetMaker(validset, transform_valid_test)
        testset = DatasetMaker(testset, transform_valid_test)

        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        validloader = data.DataLoader(validset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


        # Model
        print("==> creating model '{}'".format(args.arch))
        if args.arch.startswith('resnext'):
            base_model = models.__dict__[args.arch](
                        cardinality=args.cardinality,
                        num_classes=num_classes,
                        depth=args.depth,
                        widen_factor=args.widen_factor,
                        dropRate=args.drop,
                    )
        elif args.arch.startswith('densenet'):
            base_model = models.__dict__[args.arch](
                        num_classes=num_classes,
                        depth=args.depth,
                        growthRate=args.growthRate,
                        compressionRate=args.compressionRate,
                        dropRate=args.drop,
                    )
        elif args.arch.startswith('wrn'):
            base_model = models.__dict__[args.arch](
                        num_classes=num_classes,
                        depth=args.depth,
                        widen_factor=args.widen_factor,
                        dropRate=args.drop,
                    )
        elif args.arch.endswith('resnet'):
            base_model = models.__dict__[args.arch](
                        num_classes=num_classes,
                        depth=args.depth,
                        block_name=args.block_name,
                    )
        else:
            base_model = models.__dict__[args.arch](num_classes=num_classes)

        base_model = torch.nn.DataParallel(base_model).cuda()
        cudnn.benchmark = True
        print('    Total params: %.2fM' % (sum(p.numel() for p in base_model.parameters())/1000000.0))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(base_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Resume
        title = 'cifar-' + args.arch
        if args.resume:
            # Load checkpoint.
            print('==> Loading basemodel..')
            assert os.path.isfile(args.resume + f'-{0}/model_best.pth.tar'), 'Error: no checkpoint directory found!'
            # args.checkpoint = os.path.dirname(args.resume + f'-{run}/model_best.pth.tar')
            args.checkpoint = args.resume
            checkpoint = torch.load(args.resume + f'-{0}/model_best.pth.tar')
            best_acc = checkpoint['best_acc']
            # start_epoch = checkpoint['epoch']
            base_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('==> Loading basemodel successfully.')
            # logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(args.checkpoint + f'-{run}', 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Test Loss', 'Train Acc.', 'Valid Acc.', 'Test Acc'])


        if args.evaluate:
            print('\nEvaluation only')
            valid_loss, valid_acc = test(validloader, base_model, criterion, start_epoch, use_cuda)
            test_loss, test_acc = test(testloader, base_model, criterion, start_epoch, use_cuda)
            print('Best_acc: ', best_acc)
            print(' Valid Loss:  %.8f, Valid Acc:  %.2f' % (valid_loss, valid_acc))
            print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

        # Train and val
        for epoch in range(start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

            train_loss, train_acc = train(trainloader, base_model, criterion, optimizer, epoch, use_cuda)
            valid_loss, valid_acc = test(validloader, base_model, criterion, epoch, use_cuda)
            test_loss, test_acc = test(testloader, base_model, criterion, epoch, use_cuda)

            # append logger file
            logger.append([state['lr'], train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc])

            # save base_model
            is_best = valid_acc > best_acc
            best_acc = max(valid_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': base_model.state_dict(),
                    'acc': valid_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint=args.checkpoint + f'-{run}')

        logger.close()
        logger.plot()
        savefig(os.path.join(args.checkpoint + f'-{run}', 'log.eps'))

        print('Best acc: ', best_acc)
        continue



def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
