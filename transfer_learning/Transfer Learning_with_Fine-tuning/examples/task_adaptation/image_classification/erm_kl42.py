"""
@author: Yifei Ji, Junguang Jiang
@contact: jiyf990330@163.com, JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import sys
import argparse
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('../../..')
from tllib.modules.classifier import Classifier
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.data import ForeverDataIterator
from tllib.utils.logger import CompleteLogger
from archive import random_augment
from augmentations import *
from copy import deepcopy


sys.path.append('.')
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Augmentation(object):
    """
    Apply a subset of random augmentation policies from a set of random transformations
    """
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


class CutoutDefault(object):
    """
    Apply cutout transformation.
    Code taken from: https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class ModelEMA(object):
    def __init__(self, model, decay):
        self.ema = deepcopy(model)
        self.ema.cuda()
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])


class TransformFix(object):
    def __init__(self):
        self.weak = utils.get_train_transform(args.train_resizing, not args.no_hflip, args.color_jitter)
        self.strong = utils.get_train_transform(args.train_resizing, not args.no_hflip, args.color_jitter)

        self.strong.transforms.insert(0, Augmentation(random_augment())) #adding random augmentations
        self.strong.transforms.append(CutoutDefault(20)) #adding cutout

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return (weak), (strong)


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, not args.no_hflip, args.color_jitter)
    val_transform = utils.get_val_transform(args.val_resizing)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_dataset, val_dataset, num_classes = utils.get_dataset(args.data, args.root, TransformFix(),
                                                                    val_transform, args.sample_rate, args.num_samples_per_classes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    train_iter = ForeverDataIterator(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print("training dataset size: {} test dataset size: {}".format(len(train_dataset), len(val_dataset)))

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, args.pretrained)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = Classifier(backbone, num_classes, pool_layer=pool_layer, finetune=args.finetune).to(device)

    ema_model = ModelEMA(classifier, 0.997)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(args.lr), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epochs, gamma=args.lr_gamma)

    optimizer_bac = SGD(classifier.get_parameters(args.lr), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler_bac = torch.optim.lr_scheduler.MultiStepLR(optimizer_bac, args.lr_decay_epochs, gamma=args.lr_gamma)

    # resume from the best checkpoint
    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        acc1 = utils.validate(val_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.0
    for epoch in range(args.epochs):
        logger.set_epoch(epoch)
        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_iter, classifier, ema_model, optimizer, epoch, args)
        lr_scheduler.step()
        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

        if epoch > -1:
            train_bac(train_iter, classifier, ema_model, optimizer_bac, args)
            lr_scheduler_bac.step()

            # evaluate on validation set
            acc1 = utils.validate(val_loader, classifier, args, device)

            # remember best acc@1 and save checkpoint
            torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
            if acc1 > best_acc1:
                shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))
    logger.close()


def train_bac(train_iter: ForeverDataIterator, model: Classifier, ema_model: Classifier, optimizer_bac: SGD, args: argparse.Namespace):
    # switch to train mode
    model.train()

    all_num = 0.
    for i in range(args.iters_per_epoch):
        (x, _), _ = next(train_iter)

        x = x.to(device)
        # x_s = x_s.to(device)

        # compute output
        _, y_t = ema_model.ema(x)
        _, logits = model(x)

        pred_u_w = F.softmax(y_t, dim=1).detach().data
        pred_u_s = F.log_softmax(logits, dim=1)

        max_prob, _ = torch.max(F.softmax(y_t), dim=-1)

        # y_softmax = F.softmax(y_t, dim=1)
        loss = ( -torch.sum((pred_u_w * pred_u_s), 1) ).mean() + ( pred_u_w * pred_u_w.log() ).mean()

        all_num += max_prob.ge(0.99).float().detach().sum().item()

        # compute gradient and do SGD step
        optimizer_bac.zero_grad()
        loss.backward()
        optimizer_bac.step()

        if i % args.print_freq == 0:
            print('all_num:', all_num)


def train(train_iter: ForeverDataIterator, model: Classifier, ema_model: Classifier, optimizer: SGD,
          epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        (x, _), labels = next(train_iter)

        x = x.to(device)
        label = labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y, f = model(x)
        cls_loss = F.cross_entropy(y, label)
        loss = cls_loss

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))
        cls_acc = accuracy(y, label)[0]
        cls_accs.update(cls_acc.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ema_model.update(model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline for Finetuning')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA')
    parser.add_argument('-sr', '--sample-rate', default=100, type=int,
                        metavar='N',
                        help='sample rate of training dataset (default: 100)')
    parser.add_argument('-sc', '--num-samples-per-classes', default=None, type=int,
                        help='number of samples per classes.')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--no-hflip', action='store_true', help='no random horizontal flipping during training')
    parser.add_argument('--color-jitter', action='store_true')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--finetune', action='store_true', help='whether use 10x smaller lr for backbone')
    parser.add_argument('--pretrained', default=None,
                        help="pretrained checkpoint of the backbone. "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
    # training parameters
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N',
                        help='mini-batch size (default: 48)')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay-epochs', type=int, default=(12, ), nargs='+', help='epochs to decay lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    args = parser.parse_args()
    main(args)

