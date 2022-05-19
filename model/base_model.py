import os
import torch
from network.base_network import BaseNetwork

class BaseModel(BaseNetwork):
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt  # 参数
        self.gpu_ids = opt.gpu_ids  # GPU
        self.use_gpu = len(self.gpu_ids) > 0
        self.isTrain = opt.isTrain  # 是否要训练
        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.Tensor  # 根据是否gpu加速确定Tensor类型
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)


    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for optimer in self.optimizers:
            optimer.step()
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
