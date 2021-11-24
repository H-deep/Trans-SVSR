import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, num_channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def metric():
    net = Net()
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))


if __name__ == '__main__':
    import time
    import pynvml
    from thop import profile
    net = Net().cuda()
    start_time = time.time()
    flops, params = profile(net, (torch.ones(1, 1, 188, 620).cuda(),))
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
