import torch
import torch.nn as nn
import functools
from network.base_network import BaseNetwork


class ResnetGenerator(BaseNetwork):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 1.初始操作 [B,3,256,256] -> [B,64,256,256]
        model = [nn.ReflectionPad2d(3),  # 镜像填充 尺寸+6
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),  # 尺寸+6 通道变Ngf
                 norm_layer(ngf),
                 nn.ReLU(True)
                 ]

        n_downsampling = 2  # 下采样次数

        # 2.添加下采样操作
        for i in range(n_downsampling):
            mult = 2 ** i  # mult的遍历列表[2^0 , 2^1]
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      # (x-3+2*1)/2 + 1 = (x-1)/2 + 1
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)
                      ]

        mult = 2 ** n_downsampling  # 表现当前数据的通道数除ngf

        # 3.添加残差层
        for i in range(n_blocks):  # n_blocks 残差块数量
            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        # 输出[B,256,64,64]

        # 4.添加上采样层 --- 解码器
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        # 输出[B,64,256,256]

        # 5.添加末尾处理
        model += [nn.ReflectionPad2d(3)]  # 尺寸+6
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]  # 尺寸-6
        model += [nn.Tanh()]
        # 输出[B,3,256,256]

        # 整合
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()

        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
