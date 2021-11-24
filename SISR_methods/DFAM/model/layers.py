import torch
import torch.nn as nn


class ShiftedConv2d(nn.Module):
    def __init__(self, inc, outc):
        super(ShiftedConv2d, self).__init__()

        self.p_conv_left = ASPPModule(inc)
        self.p_conv_right = ASPPModule(inc)

        self.f_conv_left = nn.Sequential(
            nn.Conv2d(outc * 2, outc, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, 3, 1, 1, bias=False)
        )
        self.f_conv_right = nn.Sequential(
            nn.Conv2d(outc * 2, outc, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, 3, 1, 1, bias=False)
        )

        self.ca_left = ChannelAttention(channels=outc)
        self.sa_left = SpatialAttention()
        self.ca_right = ChannelAttention(channels=outc)
        self.sa_right = SpatialAttention()

    def forward(self, x_left, x_right):
        x_two = torch.cat([x_left, x_right], dim=1)  # (b, 2c, h, w)
        # _get_p
        shift_left = self.p_conv_left(x_two)  # (b, 1, h, w)
        shift_right = self.p_conv_right(x_two)  # (b, 1, h, w)
        p_left = self._get_p(shift_left)  # (b, h, w, 2)
        p_right = self._get_p(shift_right)  # (b, h, w, 2)
        # _get_x_p
        fea_shift_left = self._get_x_p(x_left, p_left)  # (b, c, h, w)
        fea_shift_right = self._get_x_p(x_right, p_right)  # (b, c, h, w)
        # fuse the stereo images
        x_left_res = self.f_conv_left(torch.cat([x_left, fea_shift_right], dim=1))
        x_right_res = self.f_conv_right(torch.cat([x_right, fea_shift_left], dim=1))
        # attention
        x_left_res = self.sa_left(self.ca_left(x_left_res))
        x_right_res = self.sa_right(self.ca_right(x_right_res))
        # shortcut
        out_left, out_right = x_left + x_left_res, x_right + x_right_res
        return out_left, out_right, p_left, p_right, shift_left, shift_right

    def _get_p(self, shift):
        dtype = shift.data.type()
        b, _, h, w = shift.size()
        # (b, h, w, 1)
        p_0_x, p_0_y = torch.meshgrid(torch.arange(0, h).type(dtype), torch.arange(0, w).type(dtype))
        p_0_x = torch.flatten(p_0_x).view(1, h, w, 1).repeat(b, 1, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, h, w, 1).repeat(b, 1, 1, 1) + shift.permute(0, 2, 3, 1)
        # (b, h, w, 2)
        p_0 = torch.cat([torch.clamp(p_0_x, 0, h - 1), torch.clamp(p_0_y, 0, w - 1)], dim=-1)
        p_0 = p_0.contiguous().round().long()
        return p_0

    def _get_x_p(self, x, p):
        b, h, w, _ = p.size()
        c = x.size(1)
        x = x.view(b, c, -1)  # (b, c, h*w)
        index = p[..., 0] * w + p[..., 1]  # p_x*w+p_y, (b, h, w)
        index = index.unsqueeze(dim=1).expand(-1, c, -1, -1).view(b, c, -1)  # (b, c, h*w)
        x_shift = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w)  # (b, c, h, w)
        return x_shift


class ASPPBlock(nn.Module):
    def __init__(self, ch=64, dilation=3):
        super(ASPPBlock, self).__init__()
        self.conv0 = nn.Conv2d(ch, ch // 2, 3, 1, 1, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(ch, ch // 2, 3, 1, padding=dilation, dilation=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch, ch, 1, 1, 0, bias=False)

    def forward(self, x):
        out0 = self.conv0(x)
        out1 = self.conv1(x)

        out = self.relu(torch.cat([out0, out1], dim=1))
        out = self.conv1x1(out)
        return out


class ASPPModule(nn.Module):
    def __init__(self, ch=64):
        super(ASPPModule, self).__init__()
        self.aspp = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, 1, 0),
            ASPPBlock(ch, 1),
            ASPPBlock(ch, 3),
            ASPPBlock(ch, 5),
            ASPPBlock(ch, 1),
            ASPPBlock(ch, 3),
            ASPPBlock(ch, 5),
            nn.Conv2d(ch, 1, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        out = self.aspp(x)
        return out


class Selective(nn.Module):
    def __init__(self, channels, reduce=4):
        super(Selective, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(channels, channels // reduce)
        self.fc2_1 = nn.Linear(channels // reduce, channels)
        self.fc2_2 = nn.Linear(channels // reduce, channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)
        features = torch.cat([x1, x2], dim=1)
        fea_u = torch.sum(features, dim=1)
        fea_s = self.gap(fea_u)
        fea_s = fea_s.squeeze(dim=3).squeeze(dim=2)
        fea_z = self.fc1(fea_s)
        vector1 = self.fc2_1(fea_z).unsqueeze_(dim=1)
        vector2 = self.fc2_2(fea_z).unsqueeze_(dim=1)
        attention_vectors = torch.cat([vector1, vector2], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (features * attention_vectors).sum(dim=1)
        return fea_v


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)


if __name__ == '__main__':

    net = ShiftedConv2d(64, 64).cuda()
    from thop import profile

    flops, params = profile(net, (torch.ones(1, 64, 128, 128).cuda(), torch.ones(1, 64, 128, 128).cuda(), ))
    total = sum([param.nelement() for param in net.parameters()])
    print('params: %.5fM' % (total / 1e6))
    print('FLOPs: %.5fGFlops' % (flops / 1e9))

    from utils import *
    get_memory_info()
