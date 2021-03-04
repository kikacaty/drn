import argparse
import json
import os
from os.path import exists, join, split


import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from torch.nn import functional as F

import data_transforms as transforms


from segment import DRNSeg, SegList, SegListMS, save_output_images, save_colorful_images, CITYSCAPE_PALETTE

from pdb import set_trace as st

target_labels = [
            5,6,7, # object: pole, traffic light, traffic sign
            11,12, # human: person, rider
            13,14,15,16,17,18 # vehicle: car, truck, bus, train, motorcycle, bicycle
        ]

target_labels = [13]

def test_seg(args):
    batch_size = args.batch_size

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()

    # optionally resume from a checkpoint
    start_epoch = 0
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

    # img = cv2.imread(args.image_path, 1)/255.
    im = Image.open(args.image_path)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    NORM_MEAN = np.array([0.29010095242892997, 0.32808144844279574, 0.28696394422942517])
    NORM_STD = np.array([0.1829540508368939, 0.18656561047509476, 0.18447508988480435])
    img = np.array(im.resize((int(im.size[0]/2),int(im.size[1]/2))))/255.
    # img = np.array(im.crop((1500,1200,2048+1500,1024+1200)))/255.
    img = img.transpose((2,0,1))
    img_input = ((img - NORM_STD.reshape((3,1,1)))/NORM_MEAN.reshape((3,1,1))).astype(np.float32)
    image = torch.from_numpy(img_input).unsqueeze(0)

    h, w = image.size()[2:4]
    image_var = Variable(image, requires_grad=False, volatile=True)

    model.eval()

    final = model(image_var)[0]
    _, pred = torch.max(final,1)
    pred = pred.cpu().data.numpy()

    pred_names = ('real_world_pred.jpg',)
    img_names = ('real_world_attack.jpg',)
    output_dir = 'real_world_output'
    save_output_images(np.moveaxis(np.expand_dims(img,0)*255,1,-1), img_names, output_dir)
    # save_output_images(np.moveaxis(adv_img,1,-1), name, output_dir)
    save_colorful_images(
        pred, pred_names, output_dir,
        CITYSCAPE_PALETTE)




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='',
                        help='Input image path')
    parser.add_argument('--arch')
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

if __name__ == '__main__':
    args = get_args()
    test_seg(args)