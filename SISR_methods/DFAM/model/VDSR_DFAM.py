import math
import torch
import torch.nn as nn
from math import sqrt
import matplotlib.pyplot as pyplot
from torchvision.transforms import ToTensor, ToPILImage
from model.layers import *


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out


class Net(nn.Module):
    def __init__(self, intervals):
        super(Net, self).__init__()
        self.n_blocks = 18
        self.intervals = intervals
        if isinstance(intervals, list):
            self.nbody = len(intervals)
        if isinstance(intervals, int):
            self.nbody = self.n_blocks // intervals

        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.residual_layer = self.make_layer(Conv_ReLU_Block, self.n_blocks)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        shifted_conv2d = []
        for _ in range(self.nbody):
            shifted_conv2d.append(ShiftedConv2d(64, 64))
        self.shifted_conv2d = nn.Sequential(*shifted_conv2d)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x_left, x_right):
        buffer_left, buffer_right = self.relu(self.input(x_left)), self.relu(self.input(x_right))
        layer = 0
        p_list = []
        shift_list = []
        for i in range(self.n_blocks):
            buffer_left, buffer_right = self.residual_layer[i](buffer_left), self.residual_layer[i](buffer_right)
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
        buffer_left, buffer_right = self.output(buffer_left), self.output(buffer_right)
        out_left, out_right = buffer_left + x_left, buffer_right + x_right

        return out_left, out_right, p_list, shift_list


if __name__ == '__main__':
    # import time
    # import pynvml
    # from thop import profile
    # net = Net([5, 10, 15]).cuda()
    # start_time = time.time()
    # flops, params = profile(net, (torch.ones(1, 1, 188, 620).cuda(), torch.ones(1, 1, 188, 620).cuda()))
    # end_time = time.time()
    # total = sum([param.nelement() for param in net.parameters()])
    # print('params: %.2fM' % (total / 1e6))
    # print('FLOPs: %.1fGFlops' % (flops / 1e9))
    # print('inference time: {:.2f}s'.format(end_time - start_time))
    # # 获取指定GPU显存信息
    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU id
    # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # print('显存占用: {:.2f} G'.format(meminfo.used / 1024 / 1024 / 1024))  # 已用显存

    import time
    import pynvml
    from thop import profile
    f = open(r'PDAM_size_log.txt', 'w')
    for size in range(100, 3000, 10):
        print('size: %.2')
        net = ShiftedConv2d(64, 64).cuda()
        start_time = time.time()
        flops, params = profile(net, (torch.ones(1, 64, 128, 128).cuda(), torch.ones(1, 64, 128, 128).cuda()))
        end_time = time.time()
        total = sum([param.nelement() for param in net.parameters()])
        print('params: %.5fM' % (total / 1e6), file=f)
        print('FLOPs: %.5fGFlops' % (flops / 1e9), file=f)
        print('inference time: {:.5f}s'.format(end_time - start_time), file=f)
        # 获取指定GPU显存信息
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU id
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print('显存占用: {:.5f} G'.format(meminfo.used / 1024 / 1024 / 1024), file=f)  # 已用显存

    f.close()
