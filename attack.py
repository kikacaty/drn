import numpy as np
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F

import cv2

from env_var import *

from pdb import set_trace as st



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
        delta = torch.zeros_like(images, requires_grad=True)
        # delta.data = (delta.data * 2 * eps - eps) * perturb_mask

        for i in range(iters) :

            start = time.time()
            step_size  = np.max([1e-3, step_size * 0.99])
            images.requires_grad = False
            delta.requires_grad = True
            outputs = model((torch.clamp(((images*std+mean)+delta),min=0, max=1)- mean)/std)[0]

            model.zero_grad()

            # remove attack
            st()
            cost = - loss(outputs*target_mask*upper_mask, labels*2*target_mask*upper_mask) - alpha * loss(outputs*perturb_mask[:,0,:,:], u_labels*perturb_mask[:,0,:,:])

            # rap attack
            if rap:
                if target_label != None:
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
            print(i,cost)

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


def transform_adv_patch(bg_img_path, patch_pos):
    img = cv2.imread('attack_bg/attack_patch.jpg')

    height, width = 300,300
    res = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)

    pts1 = np.float32([[100,100],[200,100],[0,300],[300,300]])
    pts2 = np.float32([[1,1],[299,1],[1,299],[299,299]])

    M = cv2.getPerspectiveTransform(pts2,pts1)

    dst = cv2.warpPerspective(res,M,(300,300))

    mask_img = np.ones([300,300]) * 255
    mask_dst = cv2.warpPerspective(mask_img,M,(300,300))
    mask = (mask_dst == 255)

    # cv2.imwrite('test.png', dst)

    return dst, mask

if __name__ == '__main__':
    img = cv2.imread('attack_bg/attack_patch.jpg')

    height, width = 300,300
    res = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)

    pts1 = np.float32([[100,100],[200,100],[0,300],[300,300]])
    pts2 = np.float32([[1,1],[299,1],[1,299],[299,299]])

    M = cv2.getPerspectiveTransform(pts2,pts1)

    dst = cv2.warpPerspective(res,M,(300,300))

    mask_img = np.ones([300,300]) * 255
    mask_dst = cv2.warpPerspective(mask_img,M,(300,300))
    mask = (mask_dst == 255)

    cv2.imwrite('test.png', mask)