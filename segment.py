#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import math
import os
from os.path import exists, join, split
import threading

import time

import numpy as np
import shutil

import sys
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F


import drn
import data_transforms as transforms

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral
from scipy.ndimage.filters import gaussian_filter

from pdb import set_trace as st

try:
    from modules import batchnormsync
except ImportError:
    pass

NORM_MEAN = np.array([0.29010095242892997, 0.32808144844279574, 0.28696394422942517])
NORM_STD = np.array([0.1829540508368939, 0.18656561047509476, 0.18447508988480435])

CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


TRIPLET_PALETTE = np.asarray([
    [0, 0, 0, 255],
    [217, 83, 79, 255],
    [91, 192, 222, 255]], dtype=np.uint8)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class CRF_helper(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False, drnseg=None):
        super(CRF_helper, self).__init__()

        self.drnseg = drnseg

        channels = [3, 16, 32, 64, 128, 256, 512, 512, 512]
        self.target_middle_layer_idx = 5

        self.seg = nn.Conv2d(channels[self.target_middle_layer_idx], classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        middle = self.drnseg(x)[-1]
        middle_layer_output = middle[self.target_middle_layer_idx]
        x = self.seg(middle_layer_output)
        y = self.up(x)
        return self.softmax(y), x, y, middle

    def optim_parameters(self, memo=None):
        # for param in self.base.parameters():
        #     yield param
        for param in self.seg.parameters():
            yield param

class DRNSegCRF(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False, base_model=None):
        super(DRNSegCRF, self).__init__()

        self.drnseg = DRNSeg(model_name, classes, pretrained_model=pretrained_model,
                             pretrained=pretrained, use_torch_up=use_torch_up)
        
        if os.path.isfile(base_model):
            print("=> loading checkpoint '{}'".format(base_model))
            st()
            checkpoint = torch.load(base_model)
            self.drnseg.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(base_model))

        channels = [3, 16, 32, 64, 128, 256, 512, 512, 512]
        self.target_middle_layer_idx = 5

        self.seg = nn.Conv2d(channels[self.target_middle_layer_idx], classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        middle = self.drnseg(x)[-1]
        middle_layer_output = middle[self.target_middle_layer_idx]
        x = self.seg(middle_layer_output)
        y = self.up(x)
        return self.softmax(y), x, y, middle

    def optim_parameters(self, memo=None):
        # for param in self.base.parameters():
        #     yield param
        for param in self.seg.parameters():
            yield param

class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000, out_middle=True)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        middle = list()
        for layer in self.base:
            x = layer(x)
            middle.append(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x, y, middle

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        if self.label_list is not None:
            data.append(Image.open(
                join(self.data_dir, self.label_list[index])))
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)


class SegListMS(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = data[0].size
        if self.label_list is not None:
            data.append(Image.open(
                join(self.data_dir, self.label_list[index])))
        # data = list(self.transforms(*data))
        out_data = list(self.transforms(*data))
        ms_images = [self.transforms(data[0].resize((int(w * s), int(h * s)),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)


def validate(val_loader, model, criterion, eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if type(criterion) in [torch.nn.modules.loss.L1Loss,
                                torch.nn.modules.loss.MSELoss]:
                target = target.float()
            input = input.cuda()
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)[0]
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # losses.update(loss.data[0], input.size(0))
            losses.update(loss.data, input.size(0))
            if eval_score is not None:
                score.update(eval_score(output, target_var), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Score {score.val:.3f} ({score.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    score=score))

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

    return score.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    # return score.data[0]
    return score.data


def train(train_loader, model, criterion, optimizer, epoch,
          eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        # losses.update(loss.data[0], input.size(0))
        if eval_score is not None:
            scores.update(eval_score(output, target_var), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=scores))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train_seg_crf(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_drnseg = DRNSeg(args.arch, args.classes, None,
                          pretrained=True)
    if args.pretrained:
        single_drnseg.load_state_dict(torch.load(args.pretrained))
    drnseg = torch.nn.DataParallel(single_drnseg).cuda()
    if os.path.isfile(args.base_model):
        print("=> loading checkpoint '{}'".format(args.base_model))
        checkpoint = torch.load(args.base_model)
        drnseg.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.base_model))

    single_crf_model = CRF_helper(args.arch, args.classes, None,
                          pretrained=True, drnseg=drnseg)

    crf_model = torch.nn.DataParallel(single_crf_model).cuda()
    criterion = nn.NLLLoss2d(ignore_index=255)

    criterion.cuda()

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []
    if args.random_rotate > 0:
        t.append(transforms.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t.append(transforms.RandomScale(args.random_scale))
    t.extend([transforms.RandomCrop(crop_size),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              normalize])
    train_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'train', transforms.Compose(t),
                list_dir=args.list_dir),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'val', transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]), list_dir=args.list_dir),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )

    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(single_crf_model.optim_parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            crf_model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, crf_model, criterion, eval_score=accuracy)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        train(train_loader, crf_model, criterion, optimizer, epoch,
              eval_score=accuracy)

        # evaluate on validation set
        prec1 = validate(val_loader, crf_model, criterion, eval_score=accuracy)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = os.path.join(args.save_path, 'checkpoint_latest.pth.tar')
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': crf_model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % args.save_iter == 0:
            history_path = os.path.join(args.save_path, 'checkpoint_{:03d}.pth.tar'.format(epoch + 1))
            shutil.copyfile(checkpoint_path, history_path)

def train_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, None,
                          pretrained=True)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()
    criterion = nn.NLLLoss2d(ignore_index=255)

    criterion.cuda()

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []
    if args.random_rotate > 0:
        t.append(transforms.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t.append(transforms.RandomScale(args.random_scale))
    t.extend([transforms.RandomCrop(crop_size),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              normalize])
    train_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'train', transforms.Compose(t),
                list_dir=args.list_dir),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'val', transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]), list_dir=args.list_dir),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )

    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(single_model.optim_parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, eval_score=accuracy)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              eval_score=accuracy)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, eval_score=accuracy)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = os.path.join(args.save_path, 'checkpoint_latest.pth.tar')
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % args.save_iter == 0:
            history_path = os.path.join(args.save_path, 'checkpoint_{:03d}.pth.tar'.format(epoch + 1))
            shutil.copyfile(checkpoint_path, history_path)


def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)

def save_colorful_images_with_mask(predictions, filenames, output_dir, palettes,
     p_mask=((0,0),(0,0)), t_mask=((0,0),(0,0)),rf_mask=((0,0),(0,0))):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """

    for ind in range(len(filenames)):
        im = Image.fromarray(palettes[predictions[ind].squeeze()]).convert("RGBA")

        overlay = Image.new('RGBA', im.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        draw.rectangle(p_mask, fill=(255,0,0,70))

        # Alpha composite these two images together to obtain the desired result.
        im = Image.alpha_composite(im, overlay)

        overlay = Image.new('RGBA', im.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        draw.rectangle(t_mask, fill=(0,255,0,70))

        im = Image.alpha_composite(im, overlay)

        overlay = Image.new('RGBA', im.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        draw.rectangle(rf_mask, fill=(0,0,255,70))

        # Alpha composite these two images together to obtain the desired result.
        im = Image.alpha_composite(im, overlay).convert("RGB")

        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)

def save_colorful_images_with_pointwise_mask(predictions, filenames, output_dir, palettes,
     p_mask=None, t_mask=None,rf_mask=None):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """

    for ind in range(len(filenames)):
        im = Image.fromarray(palettes[predictions[ind].squeeze()]).convert("RGBA")

        overlay = Image.new('RGBA', im.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        p_mask = Image.fromarray(p_mask[0,0].astype(np.uint8)*255, mode='L')
        draw.bitmap((0, 0), p_mask, fill=(255,0,0,70))

        # Alpha composite these two images together to obtain the desired result.
        im = Image.alpha_composite(im, overlay)

        overlay = Image.new('RGBA', im.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        t_mask = Image.fromarray(t_mask[0].astype(np.uint8)*255, mode='L')
        draw.bitmap((0, 0), t_mask, fill=(0,255,0,70))

        im = Image.alpha_composite(im, overlay)

        if rf_mask is not None:

            overlay = Image.new('RGBA', im.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
            rf_mask = Image.fromarray(rf_mask.astype(np.uint8)*255, mode='L')
            draw.bitmap((0, 0), rf_mask, fill=(0,0,255,70))

            # Alpha composite these two images together to obtain the desired result.
            im = Image.alpha_composite(im, overlay).convert("RGB")

        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def test(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    for iter, (image, label, name) in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        image_var = Variable(image, requires_grad=False, volatile=True)
        final = model(image_var)[0]
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(
                pred, name, output_dir + '_color',
                TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
        st()
    # from pdb import set_trace as st
    # st()
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)

def measureRFSize(model, image, t_patch_pos, t_patch_size, t_rec,thres = 0):
    # measure receptive field size of the model empirically

    # measuring receptive field
    height, width = image.cpu().numpy().shape[2:]


    test = torch.ones_like(image)
    test_var = Variable(test)

    logits = torch.sum(model(test_var)[0], 1)

    # test[:,:,t_patch_pos[0]-t_patch_size/2:t_patch_pos[0]+t_patch_size/2,t_patch_pos[1]-t_patch_size/2:t_patch_pos[1]+t_patch_size/2] = 0
    test[:,:,t_rec[0][1]:t_rec[1][1],
            t_rec[0][0]:t_rec[1][0]] = 0

    test_var_p = Variable(test)

    logits_p = torch.sum(model(test_var_p)[0], 1)

    loss = nn.NLLLoss2d(ignore_index=255)


    mask_idx = np.where((logits-logits_p).cpu().data.numpy()!=0)

    h = mask_idx[1].max() - mask_idx[1].min()
    w = mask_idx[2].max() - mask_idx[2].min()

    pt_wise_mask = (np.abs((logits-logits_p).cpu().data.numpy())/np.abs((logits-logits_p).cpu().data.numpy()).max()>=thres)[0]
    coord_mask = ((mask_idx[2].min(),mask_idx[1].min()),((mask_idx[2].max(),mask_idx[1].max())))

    return h,w,coord_mask,pt_wise_mask

def measureRFSize_pt(model, image, label, target_mask,thres = 0):
    # measure receptive field size of the model empirically

    # measuring receptive field
    height, width = image.cpu().numpy().shape[2:]


    images = image.cuda()
    # t_labels = torch.ones_like(label)
    t_labels = torch.zeros_like(label)
    labels = t_labels.cuda(async=True)

    u_labels = label.cuda(async=True)

    images = torch.autograd.Variable(images)
    labels = torch.autograd.Variable(labels)
    labels = torch.autograd.Variable(u_labels)

    target_mask = torch.from_numpy(target_mask).cuda()

    # loss = nn.CrossEntropyLoss()
    loss = nn.NLLLoss2d(ignore_index=255)

    images.requires_grad = True
    outputs = model(images)[0]

    model.zero_grad()

    # cost = -loss(outputs*target_mask, labels*target_mask) #+ loss(outputs*target_mask, u_labels*target_mask)
    # cost = loss(outputs*target_mask, u_labels*target_mask)
    cost = loss(outputs*target_mask, labels*target_mask)

    cost.backward()
    grad = images.grad.cpu().numpy()
    grad = (np.abs(grad)/np.abs(grad).max()).sum(axis = 1)

    mask_idx = np.where(grad!=0)

    h = mask_idx[1].max() - mask_idx[1].min()
    w = mask_idx[2].max() - mask_idx[2].min()

    pt_wise_mask = (grad>=thres)[0]
    coord_mask = ((mask_idx[2].min(),mask_idx[1].min()),((mask_idx[2].max(),mask_idx[1].max())))

    return h,w,coord_mask,pt_wise_mask

def pgd(model, image, label, target_mask, perturb_mask, step_size = 0.1, eps=10/255., iters=10, alpha = 1e-1, beta = 2., restarts=1, target_label=None, rap=False):
    images = image.cuda()
    t_labels = torch.ones_like(label)
    labels = t_labels.cuda(async=True)

    u_labels = label.cuda(async=True)

    # images = torch.autograd.Variable(images)
    # labels = torch.autograd.Variable(labels)
    u_labels = torch.autograd.Variable(u_labels)

    upper_mask = np.zeros_like(target_mask)
    upper_mask[:,0:430,:] = 1
    upper_mask = torch.from_numpy(upper_mask).cuda()
    lower_mask = np.zeros_like(target_mask)
    lower_mask[:,430:,:] = 1
    lower_mask = torch.from_numpy(lower_mask).cuda()

    target_mask = torch.from_numpy(target_mask).cuda()
    perturb_mask = torch.from_numpy(perturb_mask).cuda()

    mean = torch.from_numpy(NORM_MEAN).float().cuda().unsqueeze(0)
    mean = mean[..., None, None]
    std = torch.from_numpy(NORM_STD).float().cuda().unsqueeze(0)
    std = std[..., None, None]

    # loss = nn.CrossEntropyLoss()
    loss = nn.NLLLoss2d(ignore_index=255)

    h_loss = houdini_loss()

    best_adv_img = [images.data, -1e8]

    ori_images = images.data * std + mean

    for j in range(restarts):
        delta = torch.rand_like(images, requires_grad=True)
        # delta = torch.zeros_like(images, requires_grad=True)
        delta.data = (delta.data * 2 * eps - eps) * perturb_mask

        for i in range(iters) :

            start = time.time()
            step_size  = np.max([1e-3, step_size * 0.99])
            images.requires_grad = False
            delta.requires_grad = True
            outputs = model((torch.clamp(((images*std+mean)+delta),min=0, max=1)- mean)/std)[0]

            model.zero_grad()

            # remove attack
            cost = - loss(outputs*target_mask*upper_mask, labels*2*target_mask*upper_mask) - alpha * loss(outputs*perturb_mask[:,0,:,:], u_labels*perturb_mask[:,0,:,:])

            # rap attack
            if rap:
                if target_label:
                    # target attack
                    cost = - loss(outputs*target_mask, labels*target_label*target_mask)
                else:
                    # untargeted attack
                    cost = loss(outputs*target_mask, u_labels*target_mask)

            # targeted attack
            # cost = -loss(outputs*target_mask, labels*target_mask) - alpha * loss(outputs*perturb_mask[:,0,:,:], u_labels*perturb_mask[:,0,:,:])
            # cost = -h_loss(outputs*target_mask, labels*target_mask) - alpha * h_loss(outputs*perturb_mask[:,0,:,:], labels*perturb_mask[:,0,:,:])

            # untargeted attack
            # cost = loss(outputs*target_mask, u_labels*target_mask)
            # cost = loss(outputs*target_mask, u_labels*target_mask) - alpha * loss(outputs*perturb_mask[:,0,:,:], labels*perturb_mask[:,0,:,:])

            # mixed attack
            # cost = -loss(outputs*target_mask, labels*2*target_mask) - loss(outputs*target_mask, labels*0*target_mask) - alpha * loss(outputs*perturb_mask[:,0,:,:], u_labels*perturb_mask[:,0,:,:])

            # print(loss(outputs*target_mask*upper_mask, labels*2*target_mask*upper_mask).data, loss(outputs*target_mask*lower_mask, labels*0*target_mask*lower_mask).data, loss(outputs*perturb_mask[:,0,:,:], u_labels*perturb_mask[:,0,:,:]).data)
            # cost = - loss(outputs*target_mask*upper_mask, labels*2*target_mask*upper_mask) - alpha * loss(outputs*perturb_mask[:,0,:,:], u_labels*perturb_mask[:,0,:,:])
            # cost = - 3*loss(outputs*target_mask*upper_mask, labels*2*target_mask*upper_mask) - loss(outputs*target_mask*lower_mask, labels*0*target_mask*lower_mask) - alpha * loss(outputs*perturb_mask[:,0,:,:], u_labels*perturb_mask[:,0,:,:])
            # cost = - loss(outputs*target_mask*upper_mask, labels*2*target_mask*upper_mask) - alpha * loss(outputs*perturb_mask[:,0,:,:], u_labels*perturb_mask[:,0,:,:])

            cost.backward()
            # print(i,cost)

            adv_images = (images*std+mean) + delta + step_size*eps*delta.grad.sign() * perturb_mask
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            delta = torch.clamp(ori_images + eta, min=0, max=1).detach_() - ori_images


            if cost.cpu().data.numpy() > best_adv_img[1]:
                best_adv_img = [delta.data, cost.cpu().data.numpy()]

    return (torch.clamp(((images*std+mean)+best_adv_img[0]),min=0, max=1)- mean)/std

def cw(model, image, label, target_mask, perturb_mask, step_size = 0.05, eps=10/255., iters=10, alpha = 0.1):
    images = image.cuda()
    # t_labels = torch.ones_like(label)
    t_labels = torch.zeros_like(label)
    labels = t_labels.cuda(async=True)

    u_labels = label.cuda(async=True)

    images = torch.autograd.Variable(images)
    labels = torch.autograd.Variable(labels)
    u_labels = torch.autograd.Variable(u_labels)

    target_mask = torch.from_numpy(target_mask).cuda()
    perturb_mask = torch.from_numpy(perturb_mask).cuda()

    mean = torch.from_numpy(NORM_MEAN).float().cuda().unsqueeze(0)
    mean = mean[..., None, None]
    std = torch.from_numpy(NORM_STD).float().cuda().unsqueeze(0)
    std = std[..., None, None]

    # loss = nn.CrossEntropyLoss()
    loss = nn.NLLLoss2d(ignore_index=255)

    best_adv_img = [images.data, -1e8]

    ori_images = images.data * std + mean

    # scaling the attack eps
    # eps *= 1.7585+3.8802

    h_loss = houdini_loss()

    for i in range(iters) :
        images.requires_grad = True
        outputs = model(images)[0]

        model.zero_grad()

        # cost = -loss(outputs*target_mask, labels*target_mask) #+ loss(outputs*target_mask, u_labels*target_mask)
        # cost = loss(outputs*target_mask, u_labels*target_mask)
        cost = - h_loss(outputs*target_mask, labels*target_mask)
        print(cost)

        cost.backward()

        adv_images = (images*std+mean) + step_size*eps*images.grad * perturb_mask / torch.max(torch.abs(images.grad * perturb_mask))
        # adv_images = (images*std+mean) + step_size*eps*images.grad / torch.max(torch.abs(images.grad))
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = ((torch.clamp(ori_images + eta, min=0, max=1) - mean)/std).detach_()

        if cost.cpu().data.numpy() > best_adv_img[1]:
            best_adv_img = [images.data, cost.cpu().data.numpy()]


    return best_adv_img[0]

class houdini_loss(nn.Module):
    def __init__(self, use_cuda=True, num_class=19, ignore_index=255):
        super(houdini_loss, self).__init__()
        # self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
        self.use_cuda = use_cuda
        self.num_class = num_class
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        pred = logits.max(1)[1].data
        target = target.data
        size = list(target.size())
        if self.ignore_index is not None:
            pred[pred == self.ignore_index] = self.num_class
            target[target == self.ignore_index] = self.num_class
        pred = torch.unsqueeze(pred, dim=1)
        target = torch.unsqueeze(target, dim=1)
        size.insert(1, self.num_class+1)
        pred_onehot = torch.zeros(size)
        target_onehot = torch.zeros(size)
        if self.use_cuda:
            pred_onehot = pred_onehot.cuda()
            target_onehot = target_onehot.cuda()
        pred_onehot = pred_onehot.scatter_(1, pred, 1).narrow(1, 0, self.num_class)

        target_onehot = target_onehot.scatter_(1, target, 1).narrow(1, 0, self.num_class)
        pred_onehot = Variable(pred_onehot)
        target_onehot = Variable(target_onehot)
        neg_log_softmax = -F.log_softmax(logits, dim=1)
        # print(logits.size())
        # print(neg_log_softmax.size())
        # print(target_onehot.size())
        twod_cross_entropy = torch.sum(neg_log_softmax*target_onehot, dim=1)
        pred_score = torch.sum(logits*pred_onehot, dim=1)
        target_score = torch.sum(logits*target_onehot, dim=1)
        mask = 0.5 + 0.5 * (((pred_score-target_score)/math.sqrt(2)).erf())
        return torch.mean(mask * twod_cross_entropy)

# remote adversarial patch attack
def attack_rap(attack_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=False, 
         pgd_steps = 0, crf_model = None,
         eval_num = 100, patch_dist = 0, ms_defense = False):

    target_labels = [
            5,6,7, # object
            11,12, # human
            13,14,15,16,17,18 # vehicle    
        ]

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    t_hist = hist.copy()
    ttl_mAP = 0.
    hist_crf = np.zeros((num_classes, num_classes))
    t_hist_crf = hist_crf.copy()
    ttl_mAP_crf = 0.
    eval_cnt = 0
    for iter, (image, label, name) in enumerate(attack_data_loader):
        print('==> Attacking image [{}/{}]...'.format(eval_cnt,eval_num))
        if eval_cnt >= eval_num:
            break
        data_time.update(time.time() - end)

        height, width = image.shape[2:]

        patch_size = (50,300)
        target_size = 300
        patch_pos = (450,1100)
        patch_pos = (520+patch_dist,1100)
        patch_rec = ((patch_pos[1] - patch_size[1]//2, patch_pos[0] - patch_size[0]//2),
            (patch_pos[1] + patch_size[1]//2, patch_pos[0] + patch_size[0]//2))
        target_pos = (380,1100)
        # target_pos = (280,1100)
        target_rec = ((target_pos[1] - target_size//2, target_pos[0] - target_size//2),
            (target_pos[1] + target_size//2, target_pos[0] + target_size//2))

        target_mask = np.zeros_like(label)
        # target_mask = np.ones_like(label)
        perturb_mask = np.zeros_like(image)
        target_mask[:,target_rec[0][1]:target_rec[1][1],
            target_rec[0][0]:target_rec[1][0]] = 1
        target_mask = (np.any([label.numpy() == id for id in target_labels],axis = 0) & (target_mask == 1)).astype(np.int8) 

        if target_mask.sum() == 0:
            print('No target, skipping...')
            continue

        eval_cnt += 1
        
        perturb_mask[:,:,patch_rec[0][1]:patch_rec[1][1],
            patch_rec[0][0]:patch_rec[1][0]] = 1
        perturb_mask = perturb_mask.astype(np.int8)

        # adding perturb area to target as well
        # to avoid spoofing new obstacle at the attack perturbation area
        loss_mask = target_mask.copy()
        # loss_mask[:,patch_rec[0][1]:patch_rec[1][1],
        #     patch_rec[0][0]:patch_rec[1][0]] = 1

        # measure receptive field
        rf_h, rf_w, rf_mask, rf_mask_bit = measureRFSize_pt(model, image, label, target_mask, thres = 1.)
        # print('receptive size: ',rf_h, rf_w)

        # Tuning attack hyper params
        step_size_list = [0.1, 0.2, 0.5, 1.0]
        step_size_list = [1.0]
        eps_list = [10./255, 50./255, 100./255, 200./255]
        eps_list = [1.]

        step_size = 1.0
        eps = 1.

        # for step_size in step_size_list:
        #     print('step size: ',step_size)
        #     for eps in eps_list:
        #         print('eps: ', eps)

        if pgd_steps == 0:
            adv_image = pgd(model,image,label,loss_mask,perturb_mask, step_size = 0.1, eps=0./255, iters=1, alpha=1)
        else:
            adv_image = pgd(model,image,label,loss_mask,perturb_mask, 
            step_size = 0.1, eps=200./255, iters=pgd_steps, alpha=1, restarts=5, rap=True)

        image_var = Variable(adv_image)

        # final : log softmax
        # logits: logits
        # middle: middle layers output

        final, final_x, logits, middle = model(image_var)
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)

        adv_img = adv_image.cpu().data.numpy() * NORM_STD.reshape(1,3,1,1) + NORM_MEAN.reshape(1,3,1,1)
        adv_img *= 255

        if save_vis:
            save_output_images(pred, name, output_dir)
            save_output_images(np.moveaxis(adv_img,1,-1), name, output_dir+'_adv')
            save_colorful_images(
                pred, name, output_dir + '_color',
                TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
            if crf_model:
                save_colorful_images(
                    pred_crf, name, output_dir + '_color_CRF',
                    TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
            save_colorful_images_with_pointwise_mask(
                pred, name, output_dir + '_color_patch',
                TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE,
                p_mask = perturb_mask,
                t_mask = target_mask,
                rf_mask = rf_mask_bit)
        if has_gt:
            label_np = label.numpy()
            # changing all other labels to -1
            cur_hist = fast_hist((pred * target_mask).flatten(), (label_np*target_mask + target_mask - 1).flatten(), num_classes)
            t_hist += cur_hist
            hist += fast_hist(pred.flatten(), label_np.flatten(), num_classes)
            # idx_list = [6,7,13]
            # cur_mAP = np.nanmean(per_class_iu(cur_hist)[idx_list]) * 100
            cur_mAP = np.nanmean(per_class_iu(cur_hist)) * 100

            if math.isnan(cur_mAP):
                ttl_mAP += ttl_mAP/iter
                # st()
            else:
                ttl_mAP += cur_mAP
            logger.info('===> mAP {mAP:.3f}, avg mAP {avg_mAP:.3f}'.format(
                mAP=round(cur_mAP,2),
                avg_mAP=round(ttl_mAP/(iter+1))))
                # CRF map
            if crf_model:
                cur_hist_crf = fast_hist((pred_crf * target_mask).flatten(), (label_np*target_mask + target_mask - 1).flatten(), num_classes)
                t_hist_crf += cur_hist_crf
                hist_crf += fast_hist(pred_crf.flatten(), label_np.flatten(), num_classes)
                idx_list = [6,7,13]
                cur_mAP_crf = np.nanmean(per_class_iu(cur_hist_crf)[idx_list]) * 100
                if math.isnan(cur_mAP_crf):
                    ttl_mAP_crf += ttl_mAP_crf/iter
                    # st()
                else:
                    ttl_mAP_crf += cur_mAP_crf
                logger.info('===> CRF mAP {mAP:.3f}, avg mAP {avg_mAP:.3f}'.format(
                    mAP=round(cur_mAP_crf,2),
                    avg_mAP=round(ttl_mAP_crf/(iter+1))))
        end = time.time()
        # st()
        ious = per_class_iu(cur_hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))

        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(attack_data_loader), batch_time=batch_time,
                            data_time=data_time))


    if has_gt: #val
        ious = per_class_iu(t_hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)

def attack(attack_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=False, 
         pgd_steps = 0, crf_model = None,
         eval_num = 100, patch_dist = 0):

    target_labels = [
            5,6,7, # object
            11,12, # human
            13,14,15,16,17,18 # vehicle    
        ]

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    t_hist = hist.copy()
    ttl_mAP = 0.
    hist_crf = np.zeros((num_classes, num_classes))
    t_hist_crf = hist_crf.copy()
    ttl_mAP_crf = 0.
    eval_cnt = 0
    for iter, (image, label, name) in enumerate(attack_data_loader):
        if eval_cnt >= eval_num:
            break
        print('==> Attacking image [{}/{}]...'.format(eval_cnt,eval_num))

        data_time.update(time.time() - end)

        height, width = image.shape[2:]

        patch_size = (50,300)
        target_size = 300
        patch_pos = (520+patch_dist,1100)
        patch_rec = ((patch_pos[1] - patch_size[1]//2, patch_pos[0] - patch_size[0]//2),
            (patch_pos[1] + patch_size[1]//2, patch_pos[0] + patch_size[0]//2))
        target_pos = (380,1100)
        # target_pos = (280,1100)
        target_rec = ((target_pos[1] - target_size//2, target_pos[0] - target_size//2),
            (target_pos[1] + target_size//2, target_pos[0] + target_size//2))

        target_mask = np.zeros_like(label)
        # target_mask = np.ones_like(label)
        perturb_mask = np.zeros_like(image)
        target_mask[:,target_rec[0][1]:target_rec[1][1],target_rec[0][0]:target_rec[1][0]] = 1

        target_mask = (np.any([label.numpy() == id for id in target_labels],axis = 0) & (target_mask == 1)).astype(np.int8) 
        
        if target_mask.sum() == 0:
            print('No target, skipping...')
            continue

        eval_cnt += 1
        
        perturb_mask[:,:,patch_rec[0][1]:patch_rec[1][1],
            patch_rec[0][0]:patch_rec[1][0]] = 1
        perturb_mask = perturb_mask.astype(np.int8)

        # adding perturb area to target as well
        # to avoid spoofing new obstacle at the attack perturbation area
        loss_mask = target_mask.copy()
        # loss_mask[:,patch_rec[0][1]:patch_rec[1][1],
        #     patch_rec[0][0]:patch_rec[1][0]] = 1

        # measure receptive field
        rf_h, rf_w, rf_mask, rf_mask_bit = measureRFSize_pt(model, image, label, target_mask, thres = 1.)
        # print('receptive size: ',rf_h, rf_w)

        # Tuning attack hyper params
        step_size_list = [0.1, 0.2, 0.5, 1.0]
        step_size_list = [1.0]
        eps_list = [10./255, 50./255, 100./255, 200./255]
        eps_list = [1.]

        if pgd_steps == 0:
            adv_image = pgd(model,image,label,loss_mask,perturb_mask, step_size = 0.1, eps=0./255, iters=1, alpha=1)
        else:
            adv_image = pgd(model,image,label,loss_mask,perturb_mask, step_size = 0.1, eps=200./255, iters=pgd_steps, alpha=1, restarts=5)

        image_var = Variable(adv_image)

        # final : log softmax
        # logits: logits
        # middle: middle layers output

        final, final_x, logits, middle = model(image_var)
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)

        adv_img = adv_image.cpu().data.numpy() * NORM_STD.reshape(1,3,1,1) + NORM_MEAN.reshape(1,3,1,1)
        adv_img *= 255

        def inference_ms_image(image,ms = 2, split = False):
            w, h = image.shape[2:]
            scaled_image = F.interpolate(adv_image, scale_factor=ms, mode="bilinear", align_corners=True)
            if split:
                scaled_pred = torch.zeros(adv_image.shape[0],w*ms,h*ms)
                for i in range(ms):
                    for j in range(ms):
                        image_ms = scaled_image[:,:,i*w:(i+1)*w,j*h:(j+1)*h]
                        final, final_x, logits, middle = model(image_ms)
                        scaled_pred[:,i*w:(i+1)*w,j*h:(j+1)*h] =  torch.max(final, 1)[1]
            else:
                image_ms = scaled_image
                final, final_x, logits, middle = model(image_ms)
                scaled_pred =  torch.max(final, 1)[1]
            # pred = torch.squeeze(F.interpolate(torch.unsqueeze(scaled_pred,0), size=(w,h)),0)
            pred = scaled_pred.cpu().data.numpy().astype(np.uint8)

            return pred

        ms_pred = inference_ms_image(adv_image);save_colorful_images(ms_pred, ('ms.png',), 'test',TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)

        st()

        # remapping
        def seg_parse(model, image, pred, target_labels, seg_iter):
            fg_mask = (np.any([pred == id for id in target_labels],axis = 0)).astype(np.int8) 
            fg_mask = np.expand_dims(np.concatenate([fg_mask]*3),0)
            
            for i in range(seg_iter):

                new_image = image.copy()
                fg_image = new_image * fg_mask
                bg_image = new_image - fg_image

                fg_img = fg_image * NORM_STD.reshape(1,3,1,1) + NORM_MEAN.reshape(1,3,1,1)
                fg_img *= 255
                bg_img = bg_image * NORM_STD.reshape(1,3,1,1) + NORM_MEAN.reshape(1,3,1,1)
                bg_img *= 255

                save_output_images(np.moveaxis(fg_img,1,-1), ('fg.png',), 'test')
                save_output_images(np.moveaxis(bg_img,1,-1), ('bg.png',), 'test')


                fg_image_t = torch.from_numpy(fg_image).cuda()
                bg_image_t = torch.from_numpy(bg_image).cuda()

                final_fg = model(fg_image_t)[0]
                _, pred_fg = torch.max(final_fg, 1)
                pred_fg = pred_fg.cpu().data.numpy()


                final_bg = model(bg_image_t)[0]
                _, pred_bg = torch.max(final_bg, 1)
                pred_bg = pred_bg.cpu().data.numpy()

                save_colorful_images(pred_fg, ('fg_pred.png',), 'test',TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
                save_colorful_images(pred_bg, ('bg_pred.png',), 'test',TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)

                # adding fg in bg to fg
                fg_mask = (np.any([pred_bg == id for id in target_labels],axis = 0) | fg_mask).astype(np.int8) 
            
            return fg_mask


        def crf_dense(image, output_probs, fg_mask, crf_iter=10,
            POS_W = 3,
            POS_XY_STD = 1,
            Bi_W = 4,
            Bi_XY_STD = 67,
            Bi_RGB_STD = 3):

            U = unary_from_softmax(output_probs)
            U = np.ascontiguousarray(U)

            c,h,w = output_probs.shape

            d_img = image * NORM_STD.reshape(1,3,1,1) + NORM_MEAN.reshape(1,3,1,1)
            d_img = np.ascontiguousarray(np.moveaxis(d_img[0].astype(np.uint8),0,-1))

            d = dcrf.DenseCRF2D(w, h, c)
            d.setUnaryEnergy(U)
            d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
            
            fg_feature = np.ascontiguousarray(np.moveaxis(fg_mask[0].copy().astype(np.uint8),0,-1))*255

            d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=fg_feature, compat=Bi_W)
            # pairwise_energy = create_pairwise_bilateral(sdims=(CRF_XY_STD,CRF_XY_STD), schan=(CRF_CHAN_STD,), 
            #     img=fg_mask, chdim=0)

            Q = d.inference(crf_iter)
            print('KL: ',d.klDivergence(Q) / (h*w))
            Q = np.argmax(np.array(Q),0).reshape(pred.shape)

            save_colorful_images(Q, ('pred_crf.png',), 'test',TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)


        output_probs = F.softmax(logits, dim = 1)[0].cpu().data.numpy().copy()
        print(output_probs.shape)
        # fg_mask = seg_parse(model, adv_image.cpu().data.numpy(),pred,target_labels,5)
        # crf_dense(adv_image.cpu().data.numpy(), output_probs,fg_mask)
        # save_colorful_images(pred, ('pred.png',), 'test',TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)


        def CRF_postprocess(
            POS_W = 3, 
            POS_XY_STD = 1,
            Bi_W = 4,
            Bi_XY_STD = 67,
            Bi_RGB_STD = 3,
            crf_iter = 10,
            CRF_W = 4,
            CRF_XY_STD = 67,
            CRF_CHAN_STD = 0.1,
            PRED_W = 3, smooth=0):

            # for crf_iter in [1,5,10]:
            #     for CRF_W in [10]:
            #         for CRF_CHAN_STD in [0.01]:
            crf_model.eval()

            print('Iter: {}, weight: {}, crf_chan_std: {}'.format(crf_iter, CRF_W, CRF_CHAN_STD))
            d = dcrf.DenseCRF2D(256, 128, 19)  # width, height, nlabels
            final_crf, final_crf_x, logits_crf, middle_crf = crf_model(image_var)
            resized_image = F.interpolate(image_var, size=final_crf_x.shape[2:], mode="bilinear", align_corners=True)
            normalized_final_x = (final_x - final_x.mean())/final_x.std()
            probs = F.softmax(normalized_final_x, dim = 1)

            # set unary
            U = unary_from_softmax(probs[0].cpu().data.numpy().copy())
            d.setUnaryEnergy(U) 

            _, pred_crf = torch.max(final_crf_x, 1)
            logits_img_crf = pred_crf.cpu().data.numpy().copy()

            _, pred_small = torch.max(final_x, 1)
            logits_img = pred_small.cpu().data.numpy().copy()


            # set pairwise_energy
            d_img = resized_image.cpu().data.numpy() * NORM_STD.reshape(1,3,1,1) + NORM_MEAN.reshape(1,3,1,1)
            d_img *= 255
            d_img = np.ascontiguousarray(np.moveaxis(d_img[0].astype(np.uint8),0,-1))

            # logits_img_crf = np.moveaxis(final_crf_x[0].cpu().data.numpy().copy(),0,-1)
            # logits_img_crf = gaussian_filter(logits_img_crf,sigma=smooth)
            # logits_img_crf = np.moveaxis(logits_img_crf,-1,0)

            pairwise_energy = create_pairwise_bilateral(sdims=(CRF_XY_STD,CRF_XY_STD), schan=(CRF_CHAN_STD,), img=logits_img_crf, chdim=0)
            if CRF_W > 0:
                d.addPairwiseEnergy(pairwise_energy, compat=CRF_W) 
            # pairwise_energy = create_pairwise_bilateral(sdims=(CRF_XY_STD,CRF_XY_STD), schan=(CRF_CHAN_STD,), img=final_x[0].cpu().data.numpy().copy(), chdim=0)
            pairwise_energy = create_pairwise_bilateral(sdims=(CRF_XY_STD,CRF_XY_STD), schan=(CRF_CHAN_STD,), img=logits_img, chdim=0)
            if PRED_W > 0:
                d.addPairwiseEnergy(pairwise_energy, compat=PRED_W) 

            # d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)

            # # This adds the color-dependent term, i.e. features are (x,y,r,g,b).

            # d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=d_img,
            #                     compat=10,
            #                     kernel=dcrf.DIAG_KERNEL,
            #                     normalization=dcrf.NORMALIZE_SYMMETRIC)

            d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
            d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=d_img, compat=Bi_W)

            Q = d.inference(crf_iter)
            print('KL:', d.klDivergence(Q)/np.array(Q).shape[1])

            # reverse to logits
            crf_x = torch.log(torch.from_numpy(np.array(Q).reshape([1,19,128,256])).cuda()) + torch.log(torch.exp(normalized_final_x).sum(1))
            # crf_x = torch.log(torch.from_numpy(np.array(Q).reshape([1,19,128,256])).cuda()) + torch.log(torch.exp(normalized_final_crf_x).sum(1))
            crf_y = F.softmax(F.interpolate(crf_x, size=final_crf.shape[2:], mode="bilinear", align_corners=True),dim=1)
            # crf_y = F.softmax(model.module.up(final_x),dim=1);pred_crf = np.argmax(crf_y.cpu().detach().numpy(), axis=1).reshape([1,1024,2048])
            

            # Find out the most probable class for each pixel.
            # pred_crf = np.argmax(Q, axis=0).reshape([1,128,256])
            pred_crf = np.argmax(crf_y.cpu().detach().numpy(), axis=1).reshape([1,1024,2048])

            save_colorful_images(pred_crf, name, output_dir + '_color_CRF',TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)

            label_np = label.numpy()
            cur_hist_crf = fast_hist((pred_crf * target_mask).flatten(), (label_np*target_mask + target_mask - 1).flatten(), num_classes)
            idx_list = [6,7,13]
            cur_mAP_crf = np.nanmean(per_class_iu(cur_hist_crf)[idx_list]) * 100
            logger.debug('===> CRF mAP {mAP:.3f}, avg mAP {avg_mAP:.3f}'.format(
                mAP=round(cur_mAP_crf,2),
                avg_mAP=round(ttl_mAP_crf/(iter+1))))

        if crf_model:
                
            CRF_postprocess()
            # CRF_postprocess(CRF_W = 30,PRED_W=0,Bi_RGB_STD = 60,Bi_W=1,POS_W=0)
            st()

        if save_vis:
            save_output_images(pred, name, output_dir)
            save_output_images(np.moveaxis(adv_img,1,-1), name, output_dir+'_adv')
            save_colorful_images(
                pred, name, output_dir + '_color',
                TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
            if crf_model:
                save_colorful_images(
                    pred_crf, name, output_dir + '_color_CRF',
                    TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
            save_colorful_images_with_pointwise_mask(
                pred, name, output_dir + '_color_patch',
                TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE,
                p_mask = perturb_mask,
                t_mask = target_mask,
                rf_mask = rf_mask_bit)
        if has_gt:
            label_np = label.numpy()
            # changing all other labels to -1
            cur_hist = fast_hist((pred * target_mask).flatten(), (label_np*target_mask + target_mask - 1).flatten(), num_classes)
            t_hist += cur_hist
            hist += fast_hist(pred.flatten(), label_np.flatten(), num_classes)
            # idx_list = [6,7,13]
            # cur_mAP = np.nanmean(per_class_iu(cur_hist)[idx_list]) * 100
            cur_mAP = np.nanmean(per_class_iu(cur_hist)) * 100

            if math.isnan(cur_mAP):
                ttl_mAP += ttl_mAP/iter
                # st()
            else:
                ttl_mAP += cur_mAP
            logger.info('===> mAP {mAP:.3f}, avg mAP {avg_mAP:.3f}'.format(
                mAP=round(cur_mAP,2),
                avg_mAP=round(ttl_mAP/(iter+1))))
            
            ious = per_class_iu(cur_hist) * 100	
            logger.info(' '.join('{:.03f}'.format(i) for i in ious))
            # CRF map
            if crf_model:
                cur_hist_crf = fast_hist((pred_crf * target_mask).flatten(), (label_np*target_mask + target_mask - 1).flatten(), num_classes)
                t_hist_crf += cur_hist_crf
                hist_crf += fast_hist(pred_crf.flatten(), label_np.flatten(), num_classes)
                # idx_list = [6,7,13]
                # cur_mAP_crf = np.nanmean(per_class_iu(cur_hist_crf)[idx_list]) * 100
                cur_mAP = np.nanmean(per_class_iu(cur_hist)) * 100

                if math.isnan(cur_mAP_crf):
                    ttl_mAP_crf += ttl_mAP_crf/iter
                    # st()
                else:
                    ttl_mAP_crf += cur_mAP_crf
                logger.info('===> CRF mAP {mAP:.3f}, avg mAP {avg_mAP:.3f}'.format(
                    mAP=round(cur_mAP_crf,2),
                    avg_mAP=round(ttl_mAP_crf/(iter+1))))
                end = time.time()
                # st()
                logger.info('Eval: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            .format(iter, len(attack_data_loader), batch_time=batch_time,
                                    data_time=data_time))

                ious = per_class_iu(cur_hist_crf) * 100	
                logger.info(' '.join('{:.03f}'.format(i) for i in ious))

    if has_gt: #val
        ious = per_class_iu(t_hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        if crf_model:
            ious = per_class_iu(t_hist_crf) * 100
            logger.info('CRF: '+' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)

def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    # workers = [threading.Thread(target=resize_one, args=(i, j))
    #            for i in range(tensor.size(0)) for j in range(tensor.size(1))]

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    # for i in range(tensor.size(0)):
    #     for j in range(tensor.size(1)):
    #         out[i, j] = np.array(
    #             Image.fromarray(tensor_cpu[i, j]).resize(
    #                 (w, h), Image.BILINEAR))
    # out = tensor.new().resize_(*out.shape).copy_(torch.from_numpy(out))
    return out


def test_ms(eval_data_loader, model, num_classes, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    num_scales = len(scales)
    for iter, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        if has_gt:
            name = input_data[2]
            label = input_data[1]
        else:
            name = input_data[1]
        h, w = input_data[0].size()[2:4]
        images = [input_data[0]]
        images.extend(input_data[-num_scales:])
        # pdb.set_trace()
        outputs = []
        for image in images:
            image_var = Variable(image, requires_grad=False, volatile=True)
            final = model(image_var)[0]
            outputs.append(final.data)
        final = sum([resize_4d_tensor(out, w, h) for out in outputs])
        # _, pred = torch.max(torch.from_numpy(final), 1)
        # pred = pred.cpu().numpy()
        pred = final.argmax(axis=1)
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(pred, name, output_dir + '_color',
                                 CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()

    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    scales = [0.5, 0.75, 1.25, 1.5, 1.75]
    if args.ms:
        dataset = SegListMS(data_dir, phase, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), scales, list_dir=args.list_dir)
    else:
        dataset = SegList(data_dir, phase, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), list_dir=args.list_dir, out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = '{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        # from pdb import set_trace as st
        # st()
        mAP = test(test_loader, model, args.classes, save_vis=True,
                   has_gt=(phase != 'test' or args.with_gt), output_dir=out_dir)
    logger.info('mAP: %f', mAP)


def attack_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()

    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    scales = [0.5, 0.75, 1.25, 1.5, 1.75]
    if args.ms:
        dataset = SegListMS(data_dir, phase, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), scales, list_dir=args.list_dir)
    else:
        dataset = SegList(data_dir, phase, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), list_dir=args.list_dir, out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # crf model
    if args.crf_model:
        single_crf_model = CRF_helper(args.arch, args.classes, pretrained_model=None,
                          pretrained=False,drnseg=model)
        crf_model = torch.nn.DataParallel(single_crf_model).cuda()
        if os.path.isfile(args.crf_model):
            logger.info("=> loading checkpoint '{}'".format(args.crf_model))
            checkpoint = torch.load(args.crf_model)
            crf_model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.crf_model, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.crf_model))
    else:
        crf_model = None

    output_base_path = '{}_{}_step_{}_evalnum_{}_dist_{}'.format(args.output_path, args.arch, args.pgd_steps, args.eval_num,args.patch_dist)
    print('Saving output to {}'.format(output_base_path))
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    out_dir = '{}_{:03d}_{}_attack'.format(args.arch, start_epoch, phase)
    out_dir = os.path.join(output_base_path, out_dir)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        if args.rap:
            mAP = attack_rap(test_loader, model, args.classes, save_vis=True,
                   has_gt=(phase != 'test' or args.with_gt), 
                   output_dir=out_dir, pgd_steps = args.pgd_steps, crf_model=crf_model,
                   eval_num=args.eval_num,patch_dist=args.patch_dist)
        else:
            mAP = attack(test_loader, model, args.classes, save_vis=True,
                   has_gt=(phase != 'test' or args.with_gt), 
                   output_dir=out_dir, pgd_steps = args.pgd_steps, crf_model=crf_model,
                   eval_num=args.eval_num,patch_dist=args.patch_dist)
    logger.info('mAP: %f', mAP)

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test', 'attack', 'train_crf'])
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('-l', '--list-dir', default=None,
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--base_model', default='', type=str, metavar='PATH',
                        help='path to base model checkpoint (default: none)')
    parser.add_argument('--crf_model', default='', type=str, metavar='PATH',
                        help='path to crf model checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                        help='output path for training checkpoints')
    parser.add_argument('--log_dir', default='', type=str, metavar='PATH',
                        help='log path for saving log')
    parser.add_argument('--output_path', default='', type=str, metavar='PATH',
                        help='output path for saving output image')
    parser.add_argument('--save_iter', default=1, type=int,
                        help='number of training iterations between'
                             'checkpoint history saves')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)
    parser.add_argument('--pgd-steps', default=0, type=int)
    parser.add_argument('--eval-num', default=100, type=int)
    parser.add_argument('--patch-dist', default=0, type=int)
    parser.add_argument('--rap', action='store_true')



    args = parser.parse_args()

    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    if args.bn_sync:
        drn.BatchNorm = batchnormsync.BatchNormSync

    return args


def main():
    args = parse_args()
    FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir,exist_ok=True)
        log_filename = '{}_step_{}_evalnum_{}_dist_{}.log'.format(args.arch, args.pgd_steps, args.eval_num,args.patch_dist)
        log_path = os.path.join(args.log_dir,log_filename)
        print('Saving to log: ', log_path)
        logging.basicConfig(filename=log_path, filemode='w',format=FORMAT)

    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if args.cmd == 'train':
        train_seg(args)
    elif args.cmd == 'test':
        test_seg(args)
    elif args.cmd == 'attack':
        attack_seg(args)
    elif args.cmd == 'train_crf':
        train_seg_crf(args)


if __name__ == '__main__':
    main()
