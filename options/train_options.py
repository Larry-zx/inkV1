from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='train', help='traintest')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')

        # 训练次数相关
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--continue_train', action='store_true', help='')
        self.parser.add_argument('--which_epoch', type=int, default=1, help='')
        self.parser.add_argument('--start_dec_sup', type=int, default=60,
                                 help='The iter to start decreasing lambda_sup')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000)
        self.parser.add_argument('--save_epoch_freq', type=int, default=1)

        # 展示相关
        self.parser.add_argument('--display_freq', type=int, default=1000, help='多少个iters展示一次')
        self.parser.add_argument('--update_html_freq', type=int, default=500, help='多少个iters更新一次网页')
        self.parser.add_argument('--print_freq', type=int, default=200, help='')
        # 学习率调整相关
        self.parser.add_argument('--lr', type=float, default=0.0002, help='')
        self.parser.add_argument('--lr_policy', type=str, default='lambda',
                                 help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--niter', type=int, default=60, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=60,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')

        # loss的权重
        self.parser.add_argument('--identity', type=float, default=0.5, help='重构损失的权重')
        self.parser.add_argument('--lambda_recA', type=float, default=10.0, help='AtoBtoA的循环损失的权重')
        self.parser.add_argument('--lambda_recB', type=float, default=10.0, help='BtoAtoB的循环损失的权重')
        self.parser.add_argument('--lambda_ink', type=float, default=0.05, help='水墨损失的权重')
        self.parser.add_argument('--lambda_sup', type=float, default=0.1, help='')
        self.parser.add_argument("--lambda_Geom", type=float, default=10.0, help="weight of the geometry style loss")
        self.parser.add_argument("--lambda_recog", type=float, default=10.0, help="weight of the semantic loss")

        # 其他model
        self.parser.add_argument('--use_geom', type=int, default=1, help='1or0 是否使用geom')
        self.parser.add_argument('--finetune_netGeom', type=int, default=1, help='1or0 是否微调Geom')
        self.parser.add_argument('--geom_nc', type=int, default=3, help='')

        ### semantic loss options
        self.parser.add_argument('--N_patches', type=int, default=1, help='number of patches for clip')
        self.parser.add_argument('--patch_size', type=int, default=128, help='patchsize for clip')
        self.parser.add_argument('--num_classes', type=int, default=55, help='number of classes for inception')
        self.parser.add_argument('--cos_clip', type=int, default=0, help='use cosine similarity for CLIP semantic loss')

        self.isTrain = True
