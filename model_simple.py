import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

from VSRT.basicsr.models.archs.spynet import SPyNet

# from TDResNets.opts import parse_opts
# from TDResNets.model import generate_model


# import matplotlib.pyplot as plt
# from skimage import morphology
# from torchvision import transforms

# from stn import SpatialTransformer
# from stn_2d import STN2D
# from stn_2d import Net2

from VESPCN.option import args
# from VESPCN.model.motioncompensator import make_model as make_mc
from VSRT.basicsr.models.archs.Transformer import vsrTransformer
from sofvsr import OFRnet, optical_flow_warp
# from VSRT.basicsr.models.archs.flow_warp import flow_warp
from VSRT.basicsr.models.archs.flow_warp import flow_warp



class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))

class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=1, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

# class ResNetLayer(nn.Module):
#     """
#     A ResNet layer composed by `n` blocks stacked one after the other
#     """
#     def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
#         super().__init__()
#         # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
#         downsampling = 2 if in_channels != out_channels else 1
#         self.blocks = nn.Sequential(
#             block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
#             *[block(out_channels * block.expansion, 
#                     out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
#         )

#     def forward(self, x):
#         x = self.blocks(x)
#         return x



class Net(nn.Module):
    def __init__(self, upscale_factor, spatial_dim, cfg):
        super(Net, self).__init__()
        self.cfg = cfg

        self.scale = upscale_factor
        self.is_training = 1
        # self.OFR = OFRnet(scale=upscale_factor, channels=320, cfg=self.cfg)
        self._in_ch = 3
        self._sksize = 3
        spynet_pretrained = None
        self.spynet = Flow_spynet(spynet_pretrained)
        # self.spynet = SPyNet(pretrained=spynet_pretrained)
        
        self.upscale_factor = upscale_factor
        self.init_feature = nn.Conv2d(5*64, 32, 3, 1, 1, bias=True)
        # self.pre_transform = nn.Conv3d(3, 64, 3, 1, 1, bias=True)
        # self.init_feature3 = nn.Conv2d(5*64, 32, 3, 1, 1, bias=True)
        self.middle = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        # self.middle2 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.init_feature2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        # self.init_feature22 = nn.Conv2d(32, 3, 3, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.deep_feature = RDG(G0=32, C=4, G=24, n_RDB=4)
        self.pam = PAM(32)

        self.transformer = vsrTransformer(spatial_dim)
        self.transformer2 = vsrTransformer(spatial_dim)
        # self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.res_scale = 1
        # self.conv1 = nn.Conv2d(15, 64, 3, 1, 1, bias=True)
        # self.conv2 = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv_bres1 = nn.Conv2d(32, 256, 3, 1, 1, bias=True)

        self.ResNetBottleNeck1 = ResNetBottleNeckBlock(256, 3)
        self.conv_bres2 = nn.Conv2d(32, 256, 3, 1, 1, bias=True)

        self.ResNetBottleNeck2 = ResNetBottleNeckBlock(256, 3)

        self.upscale = nn.Sequential(
            nn.Conv2d(3, 3 * upscale_factor ** 2, 1, 1, 0, bias=True),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(3, 3, 3, 1, 1, bias=True))

    def forward(self, x_left, x_right, is_training=1):

        # x_left_compensated = torch.empty(x_left.shape[0],x_left.shape[1],x_left.shape[2], x_left.shape[3],x_left.shape[4]).to(self.cfg.device)
        # x_right_compensated = torch.empty(x_right.shape[0],x_right.shape[1],x_right.shape[2], x_right.shape[3],x_right.shape[4]).to(self.cfg.device)

        # for i in range(3):
            # draft_cubel = self.sof(x_left[:,i,:,:,:])
            # draft_cuber = self.sof(x_right[:,i,:,:,:])

        #     x_left_compensated[:,i,0,:,:] = draft_cubel[:,0,:,:]
        #     x_left_compensated[:,i,1,:,:] = draft_cubel[:,1,:,:]
        #     x_left_compensated[:,i,2,:,:] = draft_cubel[:,2,:,:]
        #     x_left_compensated[:,i,3,:,:] = draft_cubel[:,3,:,:]
        #     x_left_compensated[:,i,4,:,:] = draft_cubel[:,4,:,:]

        #     x_right_compensated[:,i,0,:,:] = draft_cuber[:,0,:,:]
        #     x_right_compensated[:,i,1,:,:] = draft_cuber[:,1,:,:]
        #     x_right_compensated[:,i,2,:,:] = draft_cuber[:,2,:,:]
        #     x_right_compensated[:,i,3,:,:] = draft_cuber[:,3,:,:]
        #     x_right_compensated[:,i,4,:,:] = draft_cuber[:,4,:,:]



        b, c, t, h, w = x_left.shape

        mid_left = self.relu(self.middle(x_left[:,:,2,:,:]))
        mid_right = self.relu(self.middle(x_right[:,:,2,:,:]))


        flows = self.spynet(x_left.permute(0,2,1,3,4)) 
        flow2 = flows[0].contiguous().view(-1, 2,h, w).permute(0, 2, 3, 1)         # [B*5, 64, 64, 2]
        x_left1 = flow_warp(x_left.view(-1, c, h, w), flow2)                                 # [B*5, 64, 64, 64]
        x_left1 = x_left1.view(b, t, c, h, w)


        flows4 = self.spynet(x_right.permute(0,2,1,3,4)) 
        flow3 = flows4[0].contiguous().view(-1, 2,h, w).permute(0, 2, 3, 1)         # [B*5, 64, 64, 2]
        x_right1 = flow_warp(x_right.view(-1, c, h, w), flow3)                                 # [B*5, 64, 64, 64]
        x_right1 = x_right1.view(b, t, c, h, w)

        # buffer_left = self.pre_transform(x_left)
        # buffer_right = self.pre_transform(x_right)
        buffer_left = self.transformer(x_left1)
        buffer_right = self.transformer2(x_right1)

        buffer_left = self.relu(self.init_feature(buffer_left))
        buffer_right = self.relu(self.init_feature(buffer_right))

        buffer_left, catfea_left = self.deep_feature(buffer_left)
        buffer_right, catfea_right = self.deep_feature(buffer_right)


        buffer_leftT, buffer_rightT = self.pam(buffer_left, buffer_right, catfea_left, catfea_right, 1)


        buffer_leftT = mid_left + buffer_leftT
        buffer_rightT = mid_right + buffer_rightT

        buffer_leftT = self.relu(self.init_feature2(buffer_leftT))
        buffer_rightT = self.relu(self.init_feature2(buffer_rightT))

        # buffer_leftT = self.bn2(buffer_leftT)
        # buffer_rightT = self.bn2(buffer_rightT)
        
        buffer_leftT = self.relu(self.conv_bres1(buffer_leftT))
        buffer_rightT = self.relu(self.conv_bres2(buffer_rightT))

        buffer_leftT = self.ResNetBottleNeck1(buffer_leftT)
        buffer_rightT = self.ResNetBottleNeck2(buffer_rightT)

        buffer_leftT = self.upscale(buffer_leftT)
        buffer_rightT = self.upscale(buffer_rightT)

        ll =  F.interpolate(x_left[:,:,2,:,:], scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        rr =  F.interpolate(x_right[:,:,2,:,:], scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)



        out_l = buffer_leftT + ll
        out_r = buffer_rightT + rr
        
        return out_l, out_r



    def sof(self, x_left):
        
        x_left = x_left.view(x_left.shape[0], 1, x_left.shape[1],x_left.shape[2],x_left.shape[3])
        x = x_left.permute(0, 2, 1, 3, 4)

        b, n_frames, c, h, w = x.size()     # x: b*n*c*h*w
        idx_center = (n_frames - 1) // 2

        # motion estimation
        flow_L1 = []
        flow_L2 = []
        flow_L3 = []
        input = []

        for idx_frame in range(n_frames):
            if idx_frame != idx_center:
                input.append(torch.cat((x[:,idx_frame,:,:,:], x[:,idx_center,:,:,:]), 1))
        optical_flow_L1, optical_flow_L2 = self.OFR(torch.cat(input, 0))

        optical_flow_L1 = optical_flow_L1.view(-1, b, 2, h//2, w//2)
        optical_flow_L2 = optical_flow_L2.view(-1, b, 2, h, w)
        # optical_flow_L3 = optical_flow_L3.view(-1, b, 2, h*self.scale, w*self.scale)

        # motion compensation
        draft_cube = []
        

        for idx_frame in range(n_frames):
            if idx_frame == idx_center:
                flow_L1.append([])
                flow_L2.append([])
                # flow_L3.append([])
                draft_cube.append(x[:, idx_center, :, :, :])
            if idx_frame != idx_center:
                if idx_frame < idx_center:
                    idx = idx_frame
                if idx_frame > idx_center:
                    idx = idx_frame - 1

                flow_L1.append(optical_flow_L1[idx, :, :, :, :])
                flow_L2.append(optical_flow_L2[idx, :, :, :, :])
                # flow_L3.append(optical_flow_L3[idx, :, :, :, :])

                for i in range(1):
                    for j in range(1):
                        draft = optical_flow_warp(x[:, idx_frame, :, :, :],
                                                  optical_flow_L2[idx, :, :, i::1, j::1] / 1)
                        draft_cube.append(draft)
        draft_cube = torch.cat(draft_cube, 1)
        return draft_cube


class Flow_spynet(nn.Module):
    def __init__(self, spynet_pretrained=None):
        super(Flow_spynet, self).__init__()
        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)
        
    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.
        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """
        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True
    
    def forward(self, lrs):
        """Compute optical flow using SPyNet for feature warping.
        Note that if the input is an mirror-extended sequence, 'flows_forward' is not needed, since it is equal to 'flows_backward.flip(1)'.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """
        n, t, c, h, w = lrs.size()    
        # assert h >= 64 and w >= 64, ('The height and width of inputs should be at least 64, 'f'but got {h} and {w}.')
        
        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        lrs_1 = torch.cat([lrs[:, 0, :, :, :].unsqueeze(1), lrs], dim=1).reshape(-1, c, h, w)  # [b*6, 3, 64, 64]
        lrs_2 = torch.cat([lrs, lrs[:, t-1, :, :, :].unsqueeze(1)], dim=1).reshape(-1, c, h, w)  # [b*6, 3, 64, 64]
        
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t+1, 2, h, w)         # [b, 6, 2, 64, 64]
        flows_backward = flows_backward[:, 1:, :, :, :]                          # [b, 5, 2, 64, 64]

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t+1, 2, h, w)      # [b, 6, 2, 64, 64]
            flows_forward = flows_forward[:, :-1, :, :, :]                       # [b, 5, 2, 64, 64]

        return flows_forward, flows_backward


class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)

class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out, buffer_cat



class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//16, 1, padding=0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channel//16, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x


class ResB3(nn.Module):
    def __init__(self, channels):
        super(ResB3, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.bq = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.bs = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(4 * channels)
        self.bn = nn.BatchNorm2d(4 * channels)

    def __call__(self, x_left, x_right, catfea_left, catfea_right, is_training):
        b, c0, h0, w0 = x_left.shape
        Q = self.bq(self.rb(self.bn(catfea_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))                    # (B*H) * C * Wr
        M_right_to_left = self.softmax(score)                                                   # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))                                  # (B*H) * Wr * Wl

        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
                           M_left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                                  ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)

        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c0, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c0, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c0, 1, 1)) +  x_rightT * V_right_tanh.repeat(1, c0, 1, 1)

        if is_training == 1:
            return out_left, out_right
        if is_training == 0:
            return out_left, out_right



class PAM3(nn.Module):
    def __init__(self, channels):
        super(PAM3, self).__init__()
        self.bq = nn.Conv3d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.bs = nn.Conv3d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(4 * channels)
        self.bn = nn.BatchNorm3d(4 * channels)

    def __call__(self, x_left, x_right, catfea_left, catfea_right, is_training):
        b, c0, h0, w0 = x_left.shape
        Q = self.bq(self.rb(self.bn(catfea_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))                    # (B*H) * C * Wr
        M_right_to_left = self.softmax(score)                                                   # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))                                  # (B*H) * Wr * Wl

        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
                           M_left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                                  ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)

        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c0, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c0, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c0, 1, 1)) +  x_rightT * V_right_tanh.repeat(1, c0, 1, 1)

        if is_training == 1:
            return out_left, out_right, \
                   (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)),\
                   (V_left_tanh, V_right_tanh)
        if is_training == 0:
            return out_left, out_right


def M_Relax(M, num_pixels):
    _, u, v = M.shape
    M_list = []
    M_list.append(M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
        pad_M = pad(M[:, :-1-i, :])
        M_list.append(pad_M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
        pad_M = pad(M[:, i+1:, :])
        M_list.append(pad_M.unsqueeze(1))
    M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
    return M_relaxed


if __name__ == "__main__":
    net = Net(upscale_factor=4)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))