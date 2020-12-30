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

from torch.nn import functional as F

import data_transforms as transforms


from segment import DRNSeg, SegList, SegListMS

from pdb import set_trace as st

target_labels = [
            5,6,7, # object: pole, traffic light, traffic sign
            11,12, # human: person, rider
            13,14,15,16,17,18 # vehicle: car, truck, bus, train, motorcycle, bicycle
        ]

target_labels = [13]

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            print(name)
            if name == 'base':
                for base_name, base_module in self.model._modules['base']._modules.items():
                    print(base_name)
                    if base_module == self.feature_module:
                        target_activations, x = self.feature_extractor(x)
                    elif "avgpool" in base_name.lower():
                        x = base_module(x)
                        x = x.view(x.size(0),-1)
                    else:
                        x = base_module(x)
            elif module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x


def preprocess_image(img):
    # means = [0.485, 0.456, 0.406]
    # stds = [0.229, 0.224, 0.225]
    NORM_MEAN = np.array([0.29010095242892997, 0.32808144844279574, 0.28696394422942517])
    NORM_STD = np.array([0.1829540508368939, 0.18656561047509476, 0.18447508988480435])

    preprocessed_img = img.copy()[:, :, ::-1]
    # for i in range(3):
    #     preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
    #     preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    # preprocessed_img = \
    #     np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    # preprocessed_img = torch.from_numpy(preprocessed_img)
    # preprocessed_img.unsqueeze_(0)
    # input = preprocessed_img.requires_grad_(True)
    return input

def reverse_image(img):
    NORM_MEAN = np.array([0.29010095242892997, 0.32808144844279574, 0.28696394422942517])
    NORM_STD = np.array([0.1829540508368939, 0.18656561047509476, 0.18447508988480435])
    img = img * NORM_STD.reshape(1,3,1,1) + NORM_MEAN.reshape(1,3,1,1)
    return np.uint8(img*255)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("grad_cam/cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None, seg_mask=0):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if np.sum(seg_mask) == 0:

            if index == None:
                index = np.argmax(output.cpu().data.numpy())

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            if self.cuda:
                one_hot = torch.sum(one_hot.cuda() * output)
            else:
                one_hot = torch.sum(one_hot * output)

            self.feature_module.zero_grad()
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

            target = features[-1]
            target = target.cpu().data.numpy()[0, :]

            weights = np.mean(grads_val, axis=(2, 3))[0, :]
            cam = np.zeros(target.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * target[i, :, :]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, input.shape[2:])
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)

        else:
            # segcam
            if index == None:

                idx_lable = np.argmax(output.cpu().data.numpy(),axis=1)
                output_np = output.cpu().data.numpy()
                one_hot = np.zeros(output.size(), dtype=np.float32)
                for i in range(output.size()[-2]):
                    for j in range(output.size()[-1]):
                        target = np.argmax(output_np[0,:,i,j])
                        if target in target_labels:
                            one_hot[0,target,i,j] = 1

                one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                seg_mask = torch.from_numpy(seg_mask)

                if self.cuda:
                    one_hot = torch.sum(one_hot.cuda() * output * seg_mask.cuda())
                else:
                    one_hot = torch.sum(one_hot * output * seg_mask.cuda())

                self.feature_module.zero_grad()
                self.model.zero_grad()
                one_hot.backward(retain_graph=True)

                grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

                target = features[-1]
                target = target.cpu().data.numpy()[0, :]

                weights = np.mean(grads_val, axis=(2, 3))[0, :]
                cam = np.zeros(target.shape[1:], dtype=np.float32)

                # target = target * grads_val[0]

                for i, w in enumerate(weights):
                    cam += w * target[i, :, :]

                # cam = grads_val[0] * target
                # cam = np.mean(cam, axis=0)
                # np.flip(cam,[0,1])

                cam = np.maximum(cam, 0)
                cam = cv2.resize(cam, input.shape[-1:-3:-1])
                cam = cam - np.min(cam)
                cam = cam / np.max(cam)

        
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None, seg_mask=0):


        if np.sum(seg_mask) == 0:
            if self.cuda:
                output = self.forward(input.cuda())
            else:
                output = self.forward(input)

            if index == None:
                index = np.argmax(output.cpu().data.numpy())

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            if self.cuda:
                one_hot = torch.sum(one_hot.cuda() * output)
            else:
                one_hot = torch.sum(one_hot * output)

            # self.model.features.zero_grad()
            # self.model.classifier.zero_grad()
            one_hot.backward(retain_graph=True)

            output = input.grad.cpu().data.numpy()
            output = output[0, :, :, :]

            return output

        else:
            # segcam

            # Model output: self.softmax(y), x, y, middle

            if self.cuda:
                output = self.forward(input.cuda())
            else:
                output = self.forward(input)

            output = output[0]

            if index == None:
                index = np.argmax(output.cpu().data.numpy())

            output_np = output.cpu().data.numpy()
            one_hot = np.zeros(output.size(), dtype=np.float32)
            for i in range(output.size()[-2]):
                for j in range(output.size()[-1]):
                    target = np.argmax(output_np[0,:,i,j])
                    if target in target_labels:
                        one_hot[0,target,i,j] = 1

            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            seg_mask = torch.from_numpy(seg_mask)

            if self.cuda:
                one_hot = torch.sum(one_hot.cuda() * output * seg_mask.cuda())
            else:
                one_hot = torch.sum(one_hot * output * seg_mask)

            # self.model.features.zero_grad()
            # self.model.classifier.zero_grad()
            one_hot.backward(retain_graph=True)

            output = input.grad.cpu().data.numpy()
            output = output[0, :, :, :]

            return output



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
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('-l', '--list-dir', default=None,
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
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

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

def seg_grad_cam(args):

    phase = 'val'
    num_workers = 8

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
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

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

    return model, test_loader


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    seg_model, eval_data_loader = seg_grad_cam(args)

    grad_cam = GradCam(model=seg_model.module, feature_module=seg_model.module.base._modules['8'], \
                       target_layer_names=["2"], use_cuda=args.use_cuda)


    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # model = models.resnet50(pretrained=True)
    # grad_cam = GradCam(model=model, feature_module=model.layer4, \
    #                    target_layer_names=["2"], use_cuda=args.use_cuda)

    if os.path.exists(args.image_path):
        img = cv2.imread(args.image_path, 1)/255.
        NORM_MEAN = np.array([0.29010095242892997, 0.32808144844279574, 0.28696394422942517])
        NORM_STD = np.array([0.1829540508368939, 0.18656561047509476, 0.18447508988480435])
        img = img.transpose((2,0,1))
        img_input = ((img - NORM_STD.reshape((3,1,1)))/NORM_MEAN.reshape((3,1,1))).astype(np.float32)
        image = torch.from_numpy(img_input).unsqueeze(0)

        st()

        h, w = image.size()[2:4]
        image_var = Variable(image, requires_grad=False, volatile=True)

        # generating gradcam mask 

        input = image.requires_grad_(True)

        target_index = None
        seg_mask = np.zeros([1,19,1024,2048])
        # seg_mask[:,:,230:530,950:1250] = 1
        seg_mask[:,:,:,:] = 1
        mask = grad_cam(input, target_index, seg_mask)

        img = reverse_image(image.cpu().data.numpy())[0].transpose((1,2,0))
        
        cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        show_cam_on_image(cv_img.astype(np.float32)/255, mask)
        st()
    else:
        for iter, (image, label, name) in enumerate(eval_data_loader):
            
            h, w = image.size()[2:4]
            outputs = []
            image_var = Variable(image, requires_grad=False, volatile=True)

            # generating gradcam mask 

            input = image.requires_grad_(True)

            target_index = None
            seg_mask = np.zeros([1,19,1024,2048])
            # seg_mask[:,:,230:530,950:1250] = 1
            seg_mask[:,:,:,:] = 1
            mask = grad_cam(input, target_index, seg_mask)

            img = reverse_image(image.cpu().data.numpy())[0].transpose((1,2,0))
            
            cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            show_cam_on_image(cv_img.astype(np.float32)/255, mask)

            st()

            gb_model = GuidedBackpropReLUModel(model=seg_model.module, use_cuda=args.use_cuda)
            gb = gb_model(input, index=target_index, seg_mask = seg_mask)

            gb = gb.transpose((1, 2, 0))
            cam_mask = cv2.merge([mask, mask, mask])
            cam_gb = deprocess_image(cam_mask*gb)
            gb = deprocess_image(gb)

            cv2.imwrite('grad_cam/gb.jpg', gb)
            cv2.imwrite('grad_cam/cam_gb.jpg', cam_gb)

            st()
            
            final = model(image_var)[0]
            outputs.append(final.data)

                
            final = sum([resize_4d_tensor(out, w, h) for out in outputs])
            # _, pred = torch.max(torch.from_numpy(final), 1)
            # pred = pred.cpu().numpy()
            pred = final.argmax(axis=1)
            batch_time.update(time.time() - end)
            '''
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
            '''

        