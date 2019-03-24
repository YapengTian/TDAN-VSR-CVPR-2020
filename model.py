import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import math
import numpy as np
from modules import ConvOffset2d

class ModelFactory(object):
    
    def create_model(self, model_name):

        if model_name == 'TDAN':
            return DSW()
        else:
            raise Exception('unknown model {}'.format(model_name))


class DSW(nn.Module):
    def __init__(self):
        super(DSW, self).__init__()
        self.name = 'DSW'
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)

        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)
        # deformable
        self.cr = nn.Conv2d(128, 64, 3, padding=1, bias=True)

        self.off2d_1 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv_1 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_2 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_2 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_3 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.deconv_3 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=True)
        self.dconv = ConvOffset2d(64, 64, (3, 3), padding=(1, 1), num_deformable_groups=8)
        self.recon_lr = nn.Conv2d(64, 3, 3, padding=1, bias=True)

        self.recon_layer = self.make_layer(Res_Block, 10)

        fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding= 1, bias=True),
                       nn.ReLU()]

        self.fea_ex = nn.Sequential(*fea_ex)
        upscaling = [
            Upsampler(default_conv, 4, 64, act=False),
            nn.Conv2d(64, 3, 3, padding=1, bias=False)]

        self.up = nn.Sequential(*upscaling)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def align(self, x, x_center):
        y = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, center, :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                continue
            supp = x[:, i, :, :, :]
            fea = torch.cat([ref, supp], dim=1)
            fea = self.cr(fea)
            # feature trans
            offset1 = self.off2d_1(fea)
            fea = self.dconv_1(fea, offset1)
            offset2 = self.off2d_2(fea)
            fea = self.deconv_2(fea, offset2)
            offset3 = self.off2d_3(fea)
            fea = self.deconv_3(supp, offset3)
            offset = self.off2d(fea)
            aligned_fea = (self.dconv(fea, offset))
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)


        y = torch.cat(y, dim=1)
        return y

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):

        batch_size, num, ch, w, h = x.size()  # 5 video frames

        # center frame interpolation
        """
        center = num //2
        x_c = x[:, center, :, :]

        """
        center = num // 2
        # extract features
        y = x.view(-1, ch, w, h)
        # y = y.unsqueeze(1)
        out = self.relu(self.conv_first(y))
        x_center = x[:, center, :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)

        # align supporting frames
        lrs = self.align(out, x_center) # motion alignments
        y = lrs.view(batch_size, -1, w, h)
        # reconstruction
        fea = self.fea_ex(y)

        out = self.recon_layer(fea)
        out = self.up(out)
        return out, lrs
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        modules = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feat, 4 * n_feat, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if bn: modules.append(nn.BatchNorm2d(n_feat))
                if act: modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feat, 9 * n_feat, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if bn: modules.append(nn.BatchNorm2d(n_feat))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*modules)


class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)

        return x + res#.mul(0.1) # 0.1
