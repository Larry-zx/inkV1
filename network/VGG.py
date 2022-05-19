import torch
import torch.nn as nn
import torchvision.models as models


class Hed(nn.Module):
    def __init__(self):
        super(Hed, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16 = self.vgg16.features
        for param in self.vgg16.parameters():  # 使用已经训练好的参数 所以参数不更新
            param.requires_grad = False

        self.score_dsn1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn4 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn5 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)

        self.fuse = nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0)

        self.upsample2 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)  # 尺寸x2
        self.upsample3 = nn.ConvTranspose2d(1, 1, kernel_size=6, stride=4, padding=1)  # 尺寸x4
        self.upsample4 = nn.ConvTranspose2d(1, 1, kernel_size=12, stride=8, padding=2)  # 尺寸x8
        self.upsample5 = nn.ConvTranspose2d(1, 1, kernel_size=24, stride=16, padding=4)  # 尺寸x16

    def forward(self, x):
        # 对于cnt4的理解 可以将vgg16的features打印出来
        cnt = 1
        res = []
        for l in self.vgg16:
            x = l(x)
            # print(cnt)
            if cnt == 4:  # 此时的x是经过conv(64,64)后在relu的feature
                y = self.score_dsn1(x)  # 通道数64->1
                res += [y]
            elif cnt == 9:
                y = self.score_dsn2(x)
                y = self.upsample2(y)
                res += [y]
            elif cnt == 16:
                y = self.score_dsn3(x)
                y = self.upsample3(y)
                res += [y]
            elif cnt == 23:
                y = self.score_dsn4(x)
                y = self.upsample4(y)
                res += [y]
            elif cnt == 30:
                y = self.score_dsn5(x)
                y = self.upsample5(y)
                res += [y]
            cnt += 1
        res = self.fuse(torch.cat(res, dim=1))  # 按通道拼接后经过conv(5,1)的卷积
        return res


def gram_matrix(input):
    bath, ch, H, W = input.size()  # 这里bath=1

    features = input.view(bath, ch, H * W)  # 进行flatten操作 变成[ch,H*W]大小的矩阵

    G = torch.bmm(features, features.permute(0, 2, 1))  # 将矩阵转置后进行内积

    return G.div(bath * ch * H * W)
