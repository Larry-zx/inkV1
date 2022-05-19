import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataRoot', required=True, help='')
        self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='')
        self.parser.add_argument('--name', required=True, help='实验名称')
        self.parser.add_argument('--size', type=int, default=256, help='')
        self.parser.add_argument('--batchSize', type=int, default=16, help='')
        self.parser.add_argument('--G_mode', type=str, default='resnet_9blocks', help='')
        self.parser.add_argument('--D_mode', type=str, default='basic', help='')
        self.parser.add_argument('--ngf', type=int, default=64, help='')
        self.parser.add_argument('--ndf', type=int, default=64, help='')
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='')
        self.parser.add_argument('--input_nc', type=int, default=3, help='')
        self.parser.add_argument('--output_nc', type=int, default=3, help='')
        self.parser.add_argument('--norm', type=str, default='instance', help='')
        self.parser.add_argument('--no_dropout', action='store_true', help='')
        self.parser.add_argument('--init_type', type=str, default='normal', help='')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='')

        # 参数文件
        self.parser.add_argument('--hed_Pth', type=str, default='checkpoints/Hed/hed.pth', help='已经训练好的HED参数文件路径')
        self.parser.add_argument('--feats2Geom_path', type=str, default='checkpoints/feats2Geom/feats2depth.pth')
        # InceptionV部分
        self.parser.add_argument('--every_feat', type=int, default=1, help='use transfer features for recog loss')

        # 展示相关
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_html', action='store_true')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        # 设置显卡id以及计算设备
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # 输出options信息
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
