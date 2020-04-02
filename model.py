import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable, Function
import math
import numpy as np
from modules import ConvOffset2d


class ModelFactory(object):

    def create_model(self, model_name):
        if model_name == 'MFSR':
            return MFSR()
        elif model_name == 'TDAN':
            return TDAN_VSR()
        elif model_name == "SISR":
            return SISR()
        else:
            raise Exception('unknown model {}'.format(model_name))

class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_align(self, weights=''):
        net_align = align_net()
        #net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_align')
            #net_align.load_state_dict(torch.load(weights))
            net_align = torch.load(weights)

            if isinstance(net_align, torch.nn.DataParallel):
                net_align = net_align.module
        return net_align

    def build_align_feat(self, weights=''):
        net_align = align_net_w_feat()
        #net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_align')
            net = torch.load(weights)
            model_dict = net_align.state_dict()
            net_dict = net.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in net_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            net_align.load_state_dict(model_dict)

            if isinstance(net_align, torch.nn.DataParallel):
                net_align = net_align.module
        return net_align

    # builder for vision
    def build_rec(self, num_block=10, weights='', scale=1.0):
        net_rec = SR_Rec(num_block, scale)

        if len(weights) > 0:
            print('Loading weights for net_rec')
            #net_rec.load_state_dict(torch.load(weights))
            net = torch.load(weights)

            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            model_dict = net_rec.state_dict()
            net_dict = net.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in net_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            net_rec.load_state_dict(model_dict)
        return net_rec

    # builder for vision
    def build_vsr_rec(self, num_block=10, weights='', scale=1.0):
        net_rec = VSR_Rec(num_block, scale)

        if len(weights) > 0:
            print('Loading weights for net_rec')
            #net_rec.load_state_dict(torch.load(weights))
            net = torch.load(weights)

            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            model_dict = net_rec.state_dict()
            net_dict = net.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in net_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            net_rec.load_state_dict(model_dict)
        return net_rec


def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class TDAN_L(nn.Module):
    def __init__(self, nets):
        super(TDAN_L, self).__init__()
        self.name = 'TDAN_L' #tdan_L with 80 blocks
        self.align_net, self.rec_net = nets

        for param in self.align_net.parameters():
            param.requires_grad = True

    def forward(self, x):

        lrs = self.align_net(x)
        y = self.rec_net(lrs)

        return y, lrs

class TDAN_F(nn.Module):
    def __init__(self, nets):
        super(TDAN_F, self).__init__()
        self.name = 'TDAN_F' #tdan_L with 80 blocks
        self.align_net, self.rec_net = nets

        for param in self.align_net.parameters():
            param.requires_grad = True

    def forward(self, x):

        lrs, feat = self.align_net(x)
        y = self.rec_net(lrs, feat)

        return y, lrs


class TDAN_VSR(nn.Module):
    def __init__(self):
        super(TDAN_VSR, self).__init__()
        self.name = 'TDAN'
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

        fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding= 1, bias=True),
                       nn.ReLU()]

        self.fea_ex = nn.Sequential(*fea_ex)
        self.recon_layer = self.make_layer(Res_Block, 10)
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
            fea = (self.dconv_1(fea, offset1))
            offset2 = self.off2d_2(fea)
            fea = (self.deconv_2(fea, offset2))
            offset3 = self.off2d_3(fea)
            fea = (self.deconv_3(supp, offset3))
            offset4 = self.off2d(fea)
            aligned_fea = (self.dconv(fea, offset4))
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

# vsr network
class TDAN(nn.Module):
    def __init__(self, nets):
        super(TDAN, self).__init__()
        self.name = 'TDAN'
        self.align_net, self.rec_net = nets

    def forward(self, x):

        lrs = self.align_net(x)
        y = self.rec_net(lrs)

        return y, lrs


# alignment network
class align_net_w_feat(nn.Module):
    def __init__(self):
        super(align_net_w_feat, self).__init__()

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

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def align(self, x, x_center):
        y = []
        feats = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, center, :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                feats.append(ref.unsqueeze(1))
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
            offset4 = self.off2d(fea)
            aligned_fea = self.dconv(fea, offset4)
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)
            feats.append(fea.unsqueeze(1))
        y = torch.cat(y, dim=1)
        feats = torch.cat(feats, dim=1)

        return y, feats

    def forward(self, x):
        batch_size, num, ch, w, h = x.size()  # 5 video frames

        # center frame interpolation
        center = num // 2

        # extract features
        y = x.view(-1, ch, w, h)
        out = self.relu(self.conv_first(y))
        x_center = x[:, center, :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)

        # align supporting frames
        lrs, feats = self.align(out, x_center)  # motion alignments
        return lrs, feats


class align_net(nn.Module):
    def __init__(self):
        super(align_net, self).__init__()

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

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

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
            offset4 = self.off2d(fea)
            aligned_fea = self.dconv(fea, offset4)
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)
        y = torch.cat(y, dim=1)
        return y

    def forward(self, x):
        batch_size, num, ch, w, h = x.size()  # 5 video frames

        # center frame interpolation
        center = num // 2

        # extract features
        y = x.view(-1, ch, w, h)
        out = self.relu(self.conv_first(y))
        x_center = x[:, center, :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)
        # align supporting frames
        lrs = self.align(out, x_center)  # motion alignments
        return lrs


# sr reconstruction network
class SR_Rec(nn.Module):
    def __init__(self, nb_block=10, scale=1.0):
        super(SR_Rec, self).__init__()
        self.recon_layer = self.make_layer(Res_Block_s(scale), nb_block)
        fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding=1, bias=True),
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

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, y):
        batch_size, num, ch, w, h = y.size()
        center = num // 2
        #y_center = y[:, center, :, :, :]
        y = y.view(batch_size, -1, w, h)
        fea = self.fea_ex(y)
        out = self.recon_layer(fea)
        out = self.up(out)
        return out #+ F.upsample(y_center, scale_factor=4, mode='bilinear')

class VSR_Rec(nn.Module):
    def __init__(self, nb_block=10, scale=1.0):
        super(VSR_Rec, self).__init__()

        fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding=1, bias=True),
                  nn.ReLU()]
        self.fea_ex = nn.Sequential(*fea_ex)
        self.fuse = nn.Conv2d(6*64, 64, 3, padding=1, bias=True)

        self.recon_layer = self.make_layer(Res_Block_s(scale), nb_block)
        upscaling = [
            Upsampler(default_conv, 4, 64, act=False),
            nn.Conv2d(64, 3, 3, padding=1, bias=False)]
        self.up = nn.Sequential(*upscaling)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, y, feats):
        batch_size, num, ch, w, h = y.size()
        center = num // 2
        #y_center = y[:, center, :, :, :]
        y = y.view(batch_size, -1, w, h)
        feat = self.fea_ex(y)
        feat = torch.cat((feats, feat.unsqueeze(1)), 1).view(batch_size, -1, w, h)
        feat = self.fuse(feat)
        out = self.recon_layer(feat)
        out = self.up(out)
        return out


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


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


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
        return x + res

class Res_Block_s(nn.Module):
    def __init__(self, scale=1.0):
        super(Res_Block_s, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.scale = scale

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res.mul(self.scale)

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
