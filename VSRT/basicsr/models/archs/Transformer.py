import torch
from torch import nn as nn
from torch.nn import functional as F

import numpy as np
from VSRT.basicsr.models.archs.arch_util import (ResidualBlockNoBN, make_layer, RCAB, ResidualGroup, default_conv, RCABWithInputConv)
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from positional_encodings import PositionalEncodingPermute3D
from torch.nn import init
import math
from torch import einsum
from VSRT.basicsr.models.archs.spynet import SPyNet, SPyNetBasicModule, ResidualBlocksWithInputConv
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from VSRT.basicsr.models.archs.flow_warp import flow_warp
import pdb



class DLA(nn.Module):
    def __init__(self, inp, oup, kernel_size = 3, stride=1, expand_ratio = 3, refine_mode='none'):
        super(DLA, self).__init__()

        hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio
        self.identity = stride == 1 and inp == oup
        self.inp, self.oup = inp, oup
        self.high_dim_id = False
        self.refine_mode = refine_mode
        self.relu = nn.ReLU6(inplace=True)
        self.bn1 = nn.BatchNorm3d(64)   

        if refine_mode == 'conv':
            self.conv = nn.Conv3d(64, 64, 3, 1, 1)

        elif refine_mode == 'conv_exapnd':
            if self.expand_ratio != 1:
                self.conv_exp = nn.Conv3d(inp, 64, 3, 1, 1)
                self.bn1 = nn.BatchNorm3d(64)   
            self.depth_sep_conv = nn.Conv3d(64, 64, 3, 1, 1)
            self.bn2 = nn.BatchNorm2d(64)

            self.conv_pro = nn.Conv3d(64, oup, 3, 1, 1)
            self.bn3 = nn.BatchNorm2d(oup)

            self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        x= input
        if self.refine_mode == 'conv':
            xx = self.conv(x)
            xx = self.relu(self.bn1(xx))
            return xx
        else:
            if self.expand_ratio !=1:
                x = self.relu(self.bn1(self.conv_exp(x)))
            x = self.relu(self.bn2(self.depth_sep_conv(x)))
            x = self.bn3(self.conv_pro(x))
            if self.identity:
                return x + input
            else:
                return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, num_feat, feat_size, fn, spatial_dim):
        super().__init__()
        self.norm = nn.LayerNorm([feat_size])
        self.fn = fn
    def forward(self, x, **kwargs):
        x = x.permute(0,1,3,4,2)
        x = self.fn(self.norm(x).permute(0,1,4,2,3), **kwargs)
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class FeedForward(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        
        self.forward_resblocks = ResidualBlocksWithInputConv(num_feat+3, num_feat, num_blocks=30)
        self.fusion = nn.Conv3d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.fusion2 = nn.Conv3d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        
    def forward(self, x, lrs=None, flows=None):
        b, t, c, h, w = x.shape

        x2 = torch.cat([x[:, 0, :, :, :].unsqueeze(1), x[:, :-1, :, :, :]], dim=1)  
        flow2 = flows[0].contiguous().view(-1, 2, h, w).permute(0, 2, 3, 1)         
        x2 = flow_warp(x2.view(-1, c, h, w), flow2)                                 
        x2 = torch.cat([lrs.contiguous().view(b*t, -1, h, w), x2], dim=1)           
        x2 = self.forward_resblocks(x2)                                             

        out = x2.view(b, c,t, h, w)      
        out = self.lrelu(self.fusion(out))   
        out = self.lrelu(self.fusion2(out))    
        out = out.view(x.shape)              

        return out


class MatmulNet(nn.Module):
    def __init__(self) -> None:
        super(MatmulNet, self).__init__()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, y)
        return x


class globalAttention(nn.Module):
    def __init__(self, num_feat=64, patch_size=8, heads=1, spatial_dim=(128,352)):
        super(globalAttention, self).__init__()
        self.heads = heads
        self.dim = patch_size ** 2 * num_feat
        self.hidden_dim = self.dim // heads
        self.num_patch = (64 // patch_size) ** 2
        
        self.scale = 64 ** -0.5
        self.DLA = DLA(self.heads,self.heads, kernel_size=3, refine_mode='conv', expand_ratio=3)
        self.to_q3 = nn.Conv3d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat) 
        self.to_k3 = nn.Conv3d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat)
        self.to_v3 = nn.Conv3d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)

        self.attend = nn.Softmax()
        self.conv = nn.Conv2d(in_channels=num_feat*3, out_channels=num_feat, kernel_size=3, padding=1)

        self.to_out = nn.Identity()
    def forward(self, x):
        b, t, c, h, w = x.shape                               
        H, D = self.heads, self.dim
        n, d = self.num_patch, self.hidden_dim 

        q = self.to_q3(x.permute(0, 2, 1, 3, 4))                       
        k = self.to_k3(x.permute(0, 2, 1, 3, 4))                     
        v = self.to_v3(x.permute(0, 2, 1, 3, 4))                    


        dots = torch.matmul(q, k.transpose(-1, -2)) * (self.dim ** -0.5)

        attn = self.attend(dots)
        attn = self.DLA(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2,1, 3, 4)
        out += x                                               

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, num_feat, feat_size, depth, patch_size, heads, spatial_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(num_feat, feat_size, globalAttention(num_feat, patch_size, heads, spatial_dim), spatial_dim)),
                Residual(PreNorm(num_feat, feat_size, FeedForward(num_feat), spatial_dim))
            ]))
            
    def forward(self, x, lrs=None, flows=None):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x, lrs=lrs, flows=flows)
        return x


class Flow_spynet(nn.Module):
    def __init__(self, spynet_pretrained=None):
        super(Flow_spynet, self).__init__()
        self.spynet = SPyNet(pretrained=spynet_pretrained)
        
    def check_if_mirror_extended(self, lrs):
        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True
    
    def forward(self, lrs):
        n, t, c, h, w = lrs.size()    
        self.check_if_mirror_extended(lrs)

        lrs_1 = torch.cat([lrs[:, 0, :, :, :].unsqueeze(1), lrs], dim=1).reshape(-1, c, h, w)  
        lrs_2 = torch.cat([lrs, lrs[:, t-1, :, :, :].unsqueeze(1)], dim=1).reshape(-1, c, h, w) 
        
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t+1, 2, h, w)        
        flows_backward = flows_backward[:, 1:, :, :, :]                         

        if self.is_mirror_extended: 
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t+1, 2, h, w)     
            flows_forward = flows_forward[:, :-1, :, :, :]                     

        return flows_forward, flows_backward


class vsrTransformer(nn.Module):
    def __init__(self,
                 spatial_dim=(32*4, 88*4),
                 image_ch=3,
                 num_feat=64,
                 feat_size=64,
                 num_frame=5, 
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 depth=20,
                 heads=30,
                 patch_size=8,
                 spynet_pretrained=None
                 ):
        super(vsrTransformer, self).__init__()
        self.num_reconstruct_block = num_reconstruct_block
        self.center_frame_idx = num_frame // 2
        self.num_frame = num_frame

        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.spynet = Flow_spynet(spynet_pretrained)

        self.pos_embedding = PositionalEncodingPermute3D(num_frame)
        self.transformer = Transformer(num_feat, feat_size, depth, patch_size, heads, spatial_dim)

        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
    def forward(self, x): 
        
        b, t, c, h, w = x.size()                                    

        feat = self.lrelu(self.conv_first(x.contiguous().view(-1, c, h, w)))      
        feat = self.feature_extraction(feat).view(b, t, -1, h, w)        

        flows = self.spynet(x)                                          
        
        feat = feat + self.pos_embedding(feat)                         
        tr_feat = self.transformer(feat, x, flows)                    
        feat = tr_feat.view(b*t, -1, h, w)                            

        feat = self.reconstruction(feat)                              
        out = self.lrelu(self.conv_hr(feat))                            
        out = out.view(b, t* out.shape[1], h, w)                                      
        
        return out

