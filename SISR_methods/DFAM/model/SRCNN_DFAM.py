import torch
import torch.nn as nn
from model.layers import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)
        self.shifted_conv2d = ShiftedConv2d(64, 64)

    def forward(self, left, right):
        p_list = []
        shift_list = []
        buffer_left, buffer_right = self.relu1(self.conv1(left)), self.relu1(self.conv1(right))
        buffer_left, buffer_right, p_left, p_right, shift_left, shift_right = \
            self.shifted_conv2d(buffer_left, buffer_right)
        p_list.append(p_left)
        p_list.append(p_right)
        shift_list.append(shift_left)
        shift_list.append(shift_right)
        buffer_left, buffer_right = self.relu2(self.conv2(buffer_left)), self.relu2(self.conv2(buffer_right))
        buffer_left, buffer_right = self.conv3(buffer_left), self.conv3(buffer_right)
        return buffer_left, buffer_right, p_list, shift_list



if __name__ == '__main__':
    import time
    import pynvml
    from thop import profile
    net = Net().cuda()
    start_time = time.time()
    flops, params = profile(net, (torch.ones(1, 1, 188, 620).cuda(), torch.ones(1, 1, 188, 620).cuda()))
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
