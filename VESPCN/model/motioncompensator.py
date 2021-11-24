import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
torch.autograd.set_detect_anomaly(True)

def make_model(args):
    return MotionCompensator(args)

class MotionCompensator(nn.Module):
    def __init__(self, args):
        self.device = 'cuda'
        if args.cpu:
            self.device = 'cpu' 
        super(MotionCompensator, self).__init__()
        print("Creating Motion compensator")

        def _gconv(in_channels, out_channels, kernel_size=3, groups=1, stride=1, bias=True):
            return nn.Conv2d(in_channels*groups, out_channels*groups, kernel_size, groups=groups, stride=stride,
                             padding=(kernel_size // 2), bias=bias)

        # Coarse flow
        coarse_flow = [_gconv(2, 24, kernel_size=5, groups=args.n_colors, stride=2), nn.ReLU()]
        coarse_flow.extend([_gconv(24, 24, kernel_size=3, groups=args.n_colors), nn.ReLU()])
        coarse_flow.extend([_gconv(24, 24, kernel_size=5, groups=args.n_colors, stride=2), nn.ReLU()])
        coarse_flow.extend([_gconv(24, 24, kernel_size=3, groups=args.n_colors), nn.ReLU()])
        coarse_flow.extend([_gconv(24, 32, kernel_size=3, groups=args.n_colors), nn.Tanh()])
        coarse_flow.extend([nn.PixelShuffle(4)])

        self.C_flow = nn.Sequential(*coarse_flow)

        # Fine flow
        fine_flow = [_gconv(5, 24, kernel_size=5, groups=args.n_colors, stride=2), nn.ReLU()]
        for _ in range(3):
            fine_flow.extend([_gconv(24, 24, kernel_size=3, groups=args.n_colors), nn.ReLU()])
        fine_flow.extend([_gconv(24, 8, kernel_size=3, groups=args.n_colors), nn.Tanh()])
        fine_flow.extend([nn.PixelShuffle(2)])

        self.F_flow = nn.Sequential(*fine_flow)

    def forward(self, frame_1, frame_2):
        # Create identity flow
        x = np.linspace(-1, 1, frame_1.shape[3])
        y = np.linspace(-1, 1, frame_1.shape[2])
        xv, yv = np.meshgrid(x, y)
        id_flow = np.expand_dims(np.stack([xv, yv], axis=-1), axis=0)
        self.identity_flow = torch.from_numpy(id_flow).float().to(self.device)

        # Coarse flow
        coarse_in = torch.cat((frame_1, frame_2), dim=1)
        coarse_out = self.C_flow(coarse_in)
        coarse_out[:,0] = coarse_out[:,0] / frame_1.shape[3]
        coarse_out[:,1] = coarse_out[:,1] / frame_2.shape[2]
        frame_2_compensated_coarse = self.warp(frame_2, coarse_out)
        
        # Fine flow
        fine_in = torch.cat((frame_1, frame_2, frame_2_compensated_coarse, coarse_out), dim=1)
        fine_out = self.F_flow(fine_in)
        fine_out[:,0] = fine_out[:,0] / frame_1.shape[3]
        fine_out[:,1] = fine_out[:,1] / frame_2.shape[2]
        flow = (coarse_out + fine_out)

        frame_2_compensated = self.warp(frame_2, flow)

        return frame_2_compensated, flow

    def warp(self, img, flow):
        # https://discuss.pytorch.org/t/solved-how-to-do-the-interpolating-of-optical-flow/5019
        # permute flow N C H W -> N H W C
        aa = self.identity_flow-flow.permute(0,2,3,1)
        # aa = aa.clamp(-1,1)
        img_compensated = F.grid_sample(img, aa, padding_mode='zeros')
        return img_compensated

    # def warp(self, img, flow):
    #     h, w = flow.shape[:2]
    #     flow = -flow
    #     flow[:,:,0] += np.arange(w)
    #     flow[:,:,1] += np.arange(h)[:,np.newaxis]
    #     res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    #     return res
    # def warp(self, x, flo):

    #     B, C, H, W = x.size()
    #     # mesh grid 
    #     xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    #     return xx

    # update at 22/08/2018 with pytorch>=0.4.0
    # def warp(x, flow, padding_mode='zeros'):
    #     """Warp an image or feature map with optical flow
    #     Args:
    #         x (Tensor): size (n, c, h, w)
    #         flow (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)
    #         padding_mode (str): 'zeros' or 'border'

    #     Returns:
    #         Tensor: warped image or feature map
    #     """
    #     assert x.size()[-2:] == flow.size()[-2:]
    #     n, _, h, w = x.size()
    #     x_ = torch.arange(w).view(1, -1).expand(h, -1)
    #     y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    #     grid = torch.stack([x_, y_], dim=0).float().cuda()
    #     grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
    #     grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    #     grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    #     grid = grid + 2 * flow
    #     grid = grid.permute(0, 2, 3, 1)
    #     return F.grid_sample(x, grid, padding_mode=padding_mode)