import numpy as np
import scipy.stats as st
import torch
from torch.optim import lr_scheduler
import torch.nn.functional as F


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


def no_sigmoid_cross_entropy(sig_logits, label):
    count_neg = torch.sum(1. - label)
    count_pos = torch.sum(label)
    beta = count_neg / (count_pos + count_neg)
    pos_weight = beta / (1 - beta)
    cost = pos_weight * label * (-1) * torch.log(sig_logits) + (1 - label) * (-1) * torch.log(1 - sig_logits)
    cost = torch.mean(cost * (1 - beta))
    return cost


def get_scheduler(optimizer, opt):
    # 学习率调整策略
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('学习率调整规则 [%s] 没有定义', opt.lr_policy)
    return scheduler



def createNRandompatches(img1, img2, N, patch_size, clipsize=224):
    myw = img1.size()[2]
    myh = img1.size()[3]

    patches1 = []
    patches2 = []

    for i in range(N):
        xcoord = int(torch.randint(myw - patch_size, ()))
        ycoord = int(torch.randint(myh - patch_size, ()))
        patch1 = img1[:, :, xcoord:xcoord+patch_size, ycoord:ycoord+patch_size]
        patches1 += [torch.nn.functional.interpolate(patch1, size=clipsize)]
        patch2 = img2[:, :, xcoord:xcoord+patch_size, ycoord:ycoord+patch_size]
        patches2 += [torch.nn.functional.interpolate(patch2, size=clipsize)]

    return patches1, patches2