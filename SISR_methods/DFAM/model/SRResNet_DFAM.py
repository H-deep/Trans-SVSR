import torch
import torch.nn as nn
from model.layers import *


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class Net(nn.Module):
    def __init__(self, n_intervals, n_blocks=16, scale=4):
        super(Net, self).__init__()
        self.n_blocks = n_blocks
        self.intervals = n_intervals
        self.scale = scale
        if isinstance(n_intervals, list):
            self.nbody = len(n_intervals)
        if isinstance(n_intervals, int):
            self.nbody = self.n_blocks // n_intervals

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        if self.scale == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif self.scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        shifted_conv2d = []
        for _ in range(self.nbody):
            shifted_conv2d.append(ShiftedConv2d(64, 64))
        self.shifted_conv2d = nn.Sequential(*shifted_conv2d)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, left, right):
        buffer_left, buffer_right = self.relu(self.conv_input(left)), self.relu(self.conv_input(right))
        residual_left, residual_right = buffer_left, buffer_right
        layer = 0
        p_list = []
        shift_list = []
        for i in range(self.n_blocks):
            buffer_left, buffer_right = self.residual[i](buffer_left), self.residual[i](buffer_right)
            if isinstance(self.intervals, list):
                if (i + 1) in self.intervals:
                    buffer_left, buffer_right, p_left, p_right, shift_left, shift_right = \
                        self.shifted_conv2d[layer](buffer_left, buffer_right)
                    p_list.append(p_left)
                    p_list.append(p_right)
                    shift_list.append(shift_left)
                    shift_list.append(shift_right)
                    layer += 1
            if isinstance(self.intervals, int):
                if (i + 1) % self.intervals == 0:
                    buffer_left, buffer_right, p_left, p_right, shift_left, shift_right = \
                        self.shifted_conv2d[layer](buffer_left, buffer_right)
                    p_list.append(p_left)
                    p_list.append(p_right)
                    shift_list.append(shift_left)
                    shift_list.append(shift_right)
                    layer += 1
        buffer_left, buffer_right = self.bn_mid(self.conv_mid(buffer_left)), self.bn_mid(self.conv_mid(buffer_right))
        buffer_left, buffer_right = torch.add(buffer_left, residual_left), torch.add(buffer_right, residual_right)
        buffer_left, buffer_right = self.upscale(buffer_left), self.upscale(buffer_right)
        out_left, out_right = self.conv_output(buffer_left), self.conv_output(buffer_right)
        return out_left, out_right, p_list, shift_list


if __name__ == '__main__':
    import time
    import pynvml
    from thop import profile
    net = Net([6, 11], scale=2).cuda()
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
