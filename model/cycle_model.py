import torch
import torch.nn as nn
import itertools
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import os

from .base_model import BaseModel
from network import define_G, define_D
from model.util import gauss_kernel, get_scheduler, no_sigmoid_cross_entropy
from network.VGG import Hed
from network.loss import GANLoss
from network.GlobalGenerator2 import GlobalGenerator2
from network.InceptionV3 import InceptionV3
from model.util import createNRandompatches
from util import util


class CycleGANModel(BaseModel):
    def __init__(self, opt):
        super(CycleGANModel, self).__init__(opt)
        self.opt = opt
        self.device = torch.device("cuda:" + str(self.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        # 定义生成器
        self.netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.G_mode, opt.norm, not opt.no_dropout,
                               opt.init_type, self.gpu_ids)
        self.netG_B = define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.G_mode, opt.norm, not opt.no_dropout,
                               opt.init_type, self.gpu_ids)
        if self.use_gpu:
            self.netG_A.cuda()
            self.netG_B.cuda()

        # 如果为训练阶段 定义辨别器
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = define_D(opt.output_nc, opt.ndf, opt.D_mode, opt.n_layers_D, opt.norm, use_sigmoid,
                                   opt.init_type, self.gpu_ids)

            self.netD_B = define_D(opt.input_nc, opt.ndf,
                                   opt.D_mode,
                                   opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            self.netD_ink = define_D(opt.output_nc, opt.ndf,
                                     opt.D_mode,
                                     opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            # 高斯核
            g_kernel = gauss_kernel(21, 3, 1).transpose((3, 2, 1, 0))
            # 卷积核21x21 参数定义为高斯核的固定参数 且取消梯度
            self.gauss_conv = nn.Conv2d(1, 1, kernel_size=21, stride=1, padding=1, bias=False)
            self.gauss_conv.weight.data.copy_(torch.from_numpy(g_kernel))
            self.gauss_conv.weight.requires_grad = False
            self.gauss_conv.cuda() if self.use_gpu else None

            # HED提取边缘
            self.HED = Hed()
            self.HED.cuda() if self.use_gpu else None
            self.HED.load_state_dict(torch.load(self.opt.hed_Pth))
            for param in self.HED.parameters():
                param.requires_grad = False

            # Geom
            if opt.use_geom == 1:
                self.netGeom = GlobalGenerator2(768, opt.geom_nc, n_downsampling=1, n_UPsampling=3)
                self.netGeom.load_state_dict(torch.load(opt.feats2Geom_path, map_location=torch.device(self.device)))
                print("记载预训练好的geom网络自 %s" % opt.feats2Geom_path)
                if opt.finetune_netGeom == 0:
                    self.netGeom.eval()

            if self.use_gpu:
                self.netD_A.cuda()
                self.netD_B.cuda()
                self.netD_ink.cuda()
                self.HED.cuda()
                self.netGeom.cuda()
        # CLIP
        ### load pretrained inception
        self.net_recog = InceptionV3(opt.num_classes, opt.isTrain, use_aux=True, pretrain=True, freeze=True,
                                     every_feat=opt.every_feat == 1)
        self.net_recog.cuda() if self.use_gpu else None
        self.net_recog.eval()
        import clip
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        clip.model.convert_weights(self.clip_model)

        # 加载已经训练好的模型
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_ink, 'D_ink', which_epoch)
                self.load_network(self.netGeom, 'Geom', which_epoch)

        # 设置优化器
        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCLIP = torch.nn.MSELoss(reduce=True)
            if opt.cos_clip == 1:
                self.criterionCLIP = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
            self.criterionGeom = torch.nn.BCELoss(reduce=True)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_ink = torch.optim.Adam(self.netD_ink.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # optim of Geom
            if (opt.use_geom == 1 and opt.finetune_netGeom == 1):
                self.optimizer_Geom = torch.optim.Adam(self.netGeom.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            self.optimizers.append(self.optimizer_D_ink)
            self.optimizers.append(self.optimizer_Geom)
            for optimizer in self.optimizers:
                self.schedulers.append(get_scheduler(optimizer, opt))

        # 输出网络
        print('---------- Networks initialized -------------')
        self.netG_A.print_network()
        self.netG_B.print_network()
        self.net_recog.print_network()
        if self.isTrain:
            self.netD_A.print_network()
            self.netD_B.print_network()
            self.netD_ink.print_network()
            self.netGeom.print_network()
        self.print_network()
        print('-----------------------------------------------')

        # 设置权重
        self.set_weights()

        # 创建文件
        if not os.path.exists(self.save_dir):
            print("创建文件夹 [%s] " % (self.save_dir))
            os.mkdir(self.save_dir)

    def set_input(self, data):
        self.input_A = data['img']
        self.input_B = data['style']
        self.A_depth = data['depth']

        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_depth = Variable(self.A_depth)

        if self.use_gpu:
            self.real_A = self.real_A.cuda()
            self.real_B = self.real_B.cuda()
            self.real_depth = self.real_depth.cuda()

        # 腐蚀操作
        kernel_size = 5
        pad_size = kernel_size // 2
        p1d = (pad_size, pad_size, pad_size, pad_size)
        p_real_B = F.pad(self.real_B, p1d, "constant", 1)
        erode_real_B = -1 * (F.max_pool2d(-1 * p_real_B, kernel_size, 1))
        erode_real_B = erode_real_B.cuda() if self.use_gpu else None

        res1 = self.gauss_conv(erode_real_B[:, 0, :, :].unsqueeze(1))
        res2 = self.gauss_conv(erode_real_B[:, 1, :, :].unsqueeze(1))
        res3 = self.gauss_conv(erode_real_B[:, 2, :, :].unsqueeze(1))
        self.ink_real_B = torch.cat((res1, res2, res3), dim=1)  # 在通道上融合 重新拼接回RGB

    def set_weights(self):
        self.lambda_idt = self.opt.identity
        self.lambda_recA = self.opt.lambda_recA
        self.lambda_recB = self.opt.lambda_recB
        self.lambda_sup = self.opt.lambda_sup  # 会更改
        self.lambda_ink = self.opt.lambda_ink
        self.lambda_Geom = self.opt.lambda_Geom
        self.lambda_recog = self.opt.lambda_recog

    # 重构损失
    def get_identityLoss(self, real_A, real_B):
        idt_A = self.netG_A(real_B)
        loss_idt_A = self.criterionIdt(idt_A, real_B) * self.lambda_recB * self.lambda_idt
        idt_B = self.netG_B(real_A)
        loss_idt_B = self.criterionIdt(idt_B, real_A) * self.lambda_recA * self.lambda_idt

        return idt_A, idt_B, loss_idt_A.item(), loss_idt_B.item()

    # 边缘损失
    def get_edgeLoss(self, real_A, fake_B):
        edge_real_A = torch.sigmoid(self.HED(real_A).detach())
        edge_fake_B = torch.sigmoid(self.HED(fake_B))
        loss_edge_1 = no_sigmoid_cross_entropy(edge_fake_B, edge_real_A) * self.lambda_sup
        return edge_real_A, edge_fake_B, loss_edge_1

    def get_G_Loss(self, fake_B, fake_A):
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)
        return loss_G_A, loss_G_B

    def get_ink_fake_B(self, fake_B):
        kernel_size = 5
        pad_size = kernel_size // 2
        p1d = (pad_size, pad_size, pad_size, pad_size)
        p_fake_B = F.pad(fake_B, p1d, "constant", 1)
        erode_fake_B = -1 * (F.max_pool2d(-1 * p_fake_B, kernel_size, 1))
        res1 = self.gauss_conv(erode_fake_B[:, 0, :, :].unsqueeze(1))
        res2 = self.gauss_conv(erode_fake_B[:, 1, :, :].unsqueeze(1))
        res3 = self.gauss_conv(erode_fake_B[:, 2, :, :].unsqueeze(1))
        ink_fake_B = torch.cat((res1, res2, res3), dim=1)
        return ink_fake_B

    def get_G_ink_loss(self, ink_fake_B):
        pred_fake_ink = self.netD_ink(ink_fake_B)
        loss_G_ink = self.criterionGAN(pred_fake_ink, True) * self.lambda_ink
        return loss_G_ink

    def get_cycle_loss(self, rec_A, rec_B):
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * self.lambda_recA
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * self.lambda_recB
        return loss_cycle_A, loss_cycle_B

    def get_geom_loss(self, geom_input, recover_geom):
        geom_input = F.interpolate(geom_input, (299, 299))  # 网络要求299x299
        recover_geom = F.interpolate(recover_geom, (304, 304))  # 预测的大小304x304
        if geom_input.size()[1] == 1:
            geom_input = geom_input.repeat(1, 3, 1, 1)
        _, geom_input = self.net_recog(geom_input)
        pred_geom = self.netGeom(geom_input)
        pred_geom = (pred_geom + 1) / 2.0
        loss_cycle_Geom = self.criterionGeom(pred_geom, recover_geom) * self.lambda_Geom
        fake_depth = F.interpolate(pred_geom, (256, 256))
        return fake_depth, loss_cycle_Geom

    def get_recog_loss(self, real_A, fake_B):
        recog_real = real_A
        recog_real0 = (recog_real[:, 0, :, :].unsqueeze(1) - 0.48145466) / 0.26862954
        recog_real1 = (recog_real[:, 1, :, :].unsqueeze(1) - 0.4578275) / 0.26130258
        recog_real2 = (recog_real[:, 2, :, :].unsqueeze(1) - 0.40821073) / 0.27577711
        recog_real = torch.cat([recog_real0, recog_real1, recog_real2], dim=1)

        line_input = fake_B
        if self.opt.output_nc == 1:
            line_input_channel0 = (line_input - 0.48145466) / 0.26862954
            line_input_channel1 = (line_input - 0.4578275) / 0.26130258
            line_input_channel2 = (line_input - 0.40821073) / 0.27577711
            line_input = torch.cat([line_input_channel0, line_input_channel1, line_input_channel2], dim=1)

        patches_r = [F.interpolate(recog_real, size=224)]  # The resize operation on tensor.
        patches_l = [F.interpolate(line_input, size=224)]

        if self.opt.N_patches > 1:
            patches_r2, patches_l2 = createNRandompatches(recog_real, line_input, self.opt.N_patches,
                                                          self.opt.patch_size)
            patches_r += patches_r2
            patches_l += patches_l2

        loss_recog = 0

        for patchnum in range(len(patches_r)):

            real_patch = patches_r[patchnum]
            line_patch = patches_l[patchnum]


            feats_r = self.clip_model.encode_image(real_patch).detach()
            feats_line = self.clip_model.encode_image(line_patch)

            myloss_recog = self.criterionCLIP(feats_line, feats_r.detach())
            if self.opt.cos_clip == 1:
                myloss_recog = 1.0 - loss_recog
                myloss_recog = torch.mean(loss_recog)

            patch_factor = (1.0 / float(self.opt.N_patches))
            if patchnum == 0:
                patch_factor = 1.0
            loss_recog += patch_factor * myloss_recog

        return loss_recog * self.lambda_recog

    def backward_G(self):

        # 重构
        idt_A, idt_B, loss_idt_A, loss_idt_B = self.get_identityLoss(self.real_A, self.real_B)
        # 生成
        fake_B = self.netG_A(self.real_A)
        fake_A = self.netG_B(self.real_B)
        # 边缘
        edge_real_A, edge_fake_B, loss_edge_1 = self.get_edgeLoss(self.real_A, fake_B)
        loss_G_A, loss_G_B = self.get_G_Loss(fake_B, fake_A)
        # 水墨
        ink_fake_B = self.get_ink_fake_B(fake_B)
        loss_G_ink = self.get_G_ink_loss(ink_fake_B)
        # 循环
        rec_A = self.netG_B(fake_B)
        rec_B = self.netG_A(fake_A)
        loss_cycle_A, loss_cycle_B = self.get_cycle_loss(rec_A, rec_B)
        # geom
        fake_depth, loss_geom = self.get_geom_loss(fake_B, self.real_depth)
        # recog
        loss_recog = self.get_recog_loss(self.real_A, fake_B)
        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_edge_1 + loss_geom + loss_recog

        loss_G.backward()

        self.fake_B = fake_B
        self.fake_A = fake_A
        self.rec_A = rec_A
        self.rec_B = rec_B
        self.edge_real_A = edge_real_A
        self.edge_fake_B = edge_fake_B
        self.ink_fake_B = ink_fake_B
        self.fake_depth = fake_depth

        self.loss_G_A = loss_G_A.item()
        self.loss_G_B = loss_G_B.item()
        self.loss_G_ink = loss_G_ink.item()
        self.loss_cycle_A = loss_cycle_A.item()
        self.loss_cycle_B = loss_cycle_B.item()
        self.loss_edge_1 = loss_edge_1.item()
        self.loss_idt_A = loss_idt_A
        self.loss_idt_B = loss_idt_B
        self.loss_geom = loss_geom.item()
        self.loss_recog = loss_recog.item()

        pass

    # 辨别器对误差回传
    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.item()
        pass

    def backward_D_B(self):
        fake_A = self.fake_A
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.item()
        pass

    def backward_D_ink(self):
        ink_fake_B = self.ink_fake_B
        loss_D_ink = self.backward_D_basic(self.netD_ink, self.ink_real_B, ink_fake_B)
        self.loss_D_ink = loss_D_ink.item()
        pass

    # 更新参数
    def optimize_parameters(self):
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
        # D_ink
        self.optimizer_D_ink.zero_grad()
        self.backward_D_ink()
        self.optimizer_D_ink.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B', self.loss_cycle_B),
                                  ('edge1', self.loss_edge_1), ('D_ink', self.loss_D_ink),
                                  ('G_ink', self.loss_G_ink), ('idt_A', self.loss_idt_A), ('idt_B', self.loss_idt_B),
                                  ('Geom', self.loss_geom), ('recog', self.loss_recog)])
        return ret_errors

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netD_ink, 'D_ink', label, self.gpu_ids)
        self.save_network(self.netGeom, 'Geom', label, self.gpu_ids)

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        edge_fake_B = util.tensor2im(self.edge_fake_B.data)
        edge_real_A = util.tensor2im(self.edge_real_A.data)
        ink_real_B = util.tensor2im(self.ink_real_B.data)
        ink_fake_B = util.tensor2im(self.ink_fake_B.data)
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),
                                   ('edge_fake_B', edge_fake_B), ('edge_real_A', edge_real_A),
                                   ('ink_real_B', ink_real_B), ('ink_fake_B', ink_fake_B)])
        return ret_visuals
