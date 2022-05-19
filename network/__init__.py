import functools
import torch.nn as nn
import torch
from network.ResnetGenerator import ResnetGenerator
from network.Discriminator import NLayerDiscriminator


def get_norm_layer(norm_type='instance'):  # 默认使用instance norm
    if norm_type == 'batch':  # batch norm
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('归一化层 [%s] 没有找到' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, model_netG, norm='batch', use_dropout=False, init_type='normal',
             gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    norm_layer = get_norm_layer(norm_type=norm)
    if model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('生成器模型 [%s] 没有被定义' % model_netG)
    if use_gpu:
        netG.cuda(gpu_ids[0])
    netG.init_weights()
    return netG


def define_D(input_nc, ndf, model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)  # 定义 归一化 层

    if use_gpu:
        assert (torch.cuda.is_available())
    if model_netD == 'basic':  # 默认使用basic
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.init_weights()
    return netD
