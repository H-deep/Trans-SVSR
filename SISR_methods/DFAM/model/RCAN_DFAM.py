import math
import torch
import torch.nn as nn
from model.layers import *


class Net(nn.Module):
    def __init__(self, intervals, scale, n_resgroups=10, n_resblocks=20, n_feats=64, reduction=16):
        super(Net, self).__init__()

        self.intervals = intervals
        if isinstance(intervals, list):
            self.nbody = len(intervals)
        if isinstance(intervals, int):
            self.nbody = self.n_resgroups // intervals

        shifted_conv2d = []
        for _ in range(self.nbody):
            shifted_conv2d.append(ShiftedConv2d(64, 64))
        self.shifted_conv2d = nn.Sequential(*shifted_conv2d)

        conv = default_conv
        n_colors = 3
        kernel_size = 3
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_range = 255
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        # modules_body = [
        #     ResidualGroup(
        #         conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
        #     for _ in range(n_resgroups)]
        modules_body = []
        for _ in range(n_resgroups):
            modules_body.append(ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks))

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x_left, x_right):
        # x = self.sub_mean(x)
        x_left, x_right = self.head(x_left), self.head(x_right)
        layer = 0
        p_list = []
        shift_list = []
        # res = self.body(x)
        buffer_left, buffer_right = x_left, x_right
        for i in range(len(self.body)):
            buffer_left, buffer_right = self.body[i](buffer_left), self.body[i](buffer_right)
            if isinstance(self.intervals, list):
                if (i+1) in self.intervals:
                    buffer_left, buffer_right, p_left, p_right, shift_left, shift_right = \
                        self.shifted_conv2d[layer](buffer_left, buffer_right)
                    p_list.append(p_left)
                    p_list.append(p_right)
                    shift_list.append(shift_left)
                    shift_list.append(shift_right)
                    layer += 1
            if isinstance(self.intervals, int):
                if (i+1) % self.intervals == 0:
                    buffer_left, buffer_right, p_left, p_right, shift_left, shift_right = \
                        self.shifted_conv2d[layer](buffer_left, buffer_right)
                    p_list.append(p_left)
                    p_list.append(p_right)
                    shift_list.append(shift_left)
                    shift_list.append(shift_right)
                    layer += 1
        buffer_left, buffer_right = buffer_left + x_left, buffer_right + x_right

        buffer_left, buffer_right = self.tail(buffer_left), self.tail(buffer_right)
        # x = self.add_mean(x)

        return buffer_left, buffer_right, p_list, shift_list


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


if __name__ == '__main__':
    import time
    import pynvml
    from thop import profile
    net = Net([4, 7], scale=2, n_resgroups=10, n_resblocks=20).cuda()
    start_time = time.time()
    flops, params = profile(net, (torch.ones(1, 3, 188, 620).cuda(), torch.ones(1, 3, 188, 620).cuda()))
    end_time = time.time()
    total = sum([param.nelement() for param in net.parameters()])
    print('params: %.2fM' % (total / 1e6))
    print('FLOPs: %.1fGFlops' % (flops / 1e9))
    print('inference time: {:.2f}s'.format(end_time - start_time))
    # 获取指定GPU显存信息
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU id
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('显存占用: {:.2f} G'.format(meminfo.used / 1024 / 1024 / 1024))  # 已用显存
