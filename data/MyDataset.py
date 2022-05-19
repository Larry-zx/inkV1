import os.path
import random
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class myDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataRoot

        self.img_dir = os.path.join(opt.dataRoot, opt.phase + '/img')
        self.depth_dir = os.path.join(opt.dataRoot, opt.phase + '/depth')
        self.style_dir = os.path.join(opt.dataRoot, opt.phase + '/style')

        self.img_paths = sorted(make_dataset(self.img_dir))
        self.depth_paths = sorted(make_dataset(self.depth_dir))
        self.style_paths = sorted(make_dataset(self.style_dir))

        self.img_len = len(self.img_paths)
        self.depth_len = len(self.depth_paths)
        self.style_len = len(self.style_paths)
        assert self.img_len == self.depth_len, "img数量与depth数量必须一致"

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        index_style = random.randint(0, self.style_len - 1)
        img_path = self.img_paths[index % self.img_len]
        depth_path = self.depth_paths[index % self.depth_len]
        style_path = self.style_paths[index_style]

        img = Image.open(img_path).convert('RGB')
        depth = Image.open(depth_path).convert('RGB')  # 单通道
        style = Image.open(style_path).convert('RGB')

        img = self.transform(img)
        depth = self.transform(depth)
        style = self.transform(style)
        out = {
            'img': img,
            'depth': depth,
            'style': style
        }
        return out

    def __len__(self):
        return self.img_len

    def name(self):
        return 'myDataset'
