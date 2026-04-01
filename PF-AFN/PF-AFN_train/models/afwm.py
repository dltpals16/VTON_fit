import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------
# AdaIN Module
# -------------------------
class AdaIN(nn.Module):
    def __init__(self, cond_dim, feature_channels):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feature_channels)
        self.beta_fc = nn.Linear(cond_dim, feature_channels)

    def forward(self, x, cond):
        B, C, H, W = x.shape
        mu = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-5
        x_norm = (x - mu) / std
        gamma = self.gamma_fc(cond).view(B, C, 1, 1)
        beta = self.beta_fc(cond).view(B, C, 1, 1)
        return gamma * x_norm + beta

class ResBlockAdaIN(nn.Module):
    def __init__(self, in_channels, cond_dim):
        super().__init__()
        self.adain1 = AdaIN(cond_dim, in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.adain2 = AdaIN(cond_dim, in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)

    def forward(self, x, cond):
        identity = x
        out = F.relu(self.adain1(x, cond))
        out = self.conv1(out)
        out = F.relu(self.adain2(out, cond))
        out = self.conv2(out)
        return out + identity

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x)

class FeatureEncoderAdaIN(nn.Module):
    def __init__(self, in_channels, cond_dim, chns=[64, 128, 256, 256, 256]):
        super().__init__()
        self.encoders = nn.ModuleList()
        for i, out_chns in enumerate(chns):
            blocks = []
            if i == 0:
                blocks.append(DownSample(in_channels, out_chns))
            else:
                blocks.append(DownSample(chns[i - 1], out_chns))
            blocks.append(ResBlockAdaIN(out_chns, cond_dim))
            blocks.append(ResBlockAdaIN(out_chns, cond_dim))
            self.encoders.append(nn.ModuleList(blocks))

    def forward(self, x, cond):
        features = []
        for down, rb1, rb2 in self.encoders:
            x = down(x)
            x = rb1(x, cond)
            x = rb2(x, cond)
            features.append(x)
        return features

class RefinePyramid(nn.Module):
    def __init__(self, chns=[64, 128, 256, 256, 256], fpn_dim=256):
        super().__init__()
        self.adaptive = nn.ModuleList([nn.Conv2d(in_c, fpn_dim, 1) for in_c in reversed(chns)])
        self.smooth = nn.ModuleList([nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1) for _ in chns])

    def forward(self, x):
        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(reversed(x)):
            feature = self.adaptive[i](conv_ftr)
            if last_feature is not None:
                feature = feature + F.interpolate(last_feature, scale_factor=2, mode='nearest')
            feature = self.smooth[i](feature)
            last_feature = feature
            feature_list.append(feature)
        return tuple(reversed(feature_list))

from .correlation import correlation
from options.train_options import TrainOptions
opt = TrainOptions().parse()

def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...] for dim, grid in enumerate(grid_list)]
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0 for grid, size in zip(grid_list, reversed(sizes))]
    return torch.stack(grid_list, dim=-1)

def TVLoss(x):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]
    return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))

class AFlowNet(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256):
        super().__init__()
        self.netMain = nn.ModuleList()
        self.netRefine = nn.ModuleList()
        for _ in range(num_pyramid):
            self.netMain.append(nn.Sequential(
                nn.Conv2d(49, 128, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(32, 2, 3, padding=1)
            ))
            self.netRefine.append(nn.Sequential(
                nn.Conv2d(2 * fpn_dim, 128, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv2d(32, 2, 3, padding=1)
            ))

        filters = torch.tensor([
            [[0, 0, 0], [1, -2, 1], [0, 0, 0]],
            [[0, 1, 0], [0, -2, 0], [0, 1, 0]],
            [[1, 0, 0], [0, -2, 0], [0, 0, 1]],
            [[0, 0, 1], [0, -2, 0], [1, 0, 0]]
        ]).unsqueeze(1).repeat(1, 1, 1, 1).float()
        self.weight = nn.Parameter(filters.cuda(), requires_grad=False)

    def forward(self, x, x_edge, x_warps, x_conds, warp_feature=True):
        last_flow = None
        last_flow_all, flow_all, delta_list = [], [], []
        x_all, x_edge_all, delta_x_all, delta_y_all = [], [], [], []
        cond_fea_all = []

        for i in range(len(x_warps)):
            x_warp = x_warps[-1 - i]
            x_cond = x_conds[-1 - i]
            cond_fea_all.append(x_cond)

            if last_flow is not None and warp_feature:
                x_warp = F.grid_sample(x_warp, last_flow.detach().permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

            corr = correlation.FunctionCorrelation(
                tenFirst=x_warp.contiguous(),
                tenSecond=x_cond.contiguous(),
                intStride=1
            )
            flow = self.netMain[i](F.leaky_relu(corr, 0.1, inplace=False))
            delta_list.append(flow)
            flow = apply_offset(flow)
            if last_flow is not None:
                flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')
            else:
                flow = flow.permute(0, 3, 1, 2)
            last_flow = flow

            x_warp = F.grid_sample(x_warp, flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
            refined = self.netRefine[i](torch.cat([x_warp, x_cond], dim=1))
            flow_all.append(refined)
            refined_flow = apply_offset(refined)
            last_flow = F.grid_sample(last_flow, refined_flow, mode='bilinear', padding_mode='border')

            last_flow = F.interpolate(last_flow, scale_factor=2, mode='bilinear')
            last_flow_all.append(last_flow)

            cur_x = F.interpolate(x, scale_factor=0.5 ** (len(x_warps) - 1 - i), mode='bilinear')
            cur_x_warp = F.grid_sample(cur_x, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
            x_all.append(cur_x_warp)

            cur_x_edge = F.interpolate(x_edge, scale_factor=0.5 ** (len(x_warps) - 1 - i), mode='bilinear')
            cur_x_warp_edge = F.grid_sample(cur_x_edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_edge_all.append(cur_x_warp_edge)

            flow_x, flow_y = torch.split(last_flow, 1, dim=1)
            delta_x = F.conv2d(flow_x, self.weight)
            delta_y = F.conv2d(flow_y, self.weight)
            delta_x_all.append(delta_x)
            delta_y_all.append(delta_y)

        x_warp = F.grid_sample(x, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
        return x_warp, last_flow, last_flow_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all

class AFWM(nn.Module):
    def __init__(self, input_nc, cond_dim=3):
        super().__init__()
        num_filters = [64, 128, 256, 256, 256]
        self.image_features = FeatureEncoderAdaIN(3, cond_dim, num_filters)
        self.cond_features = FeatureEncoderAdaIN(input_nc, cond_dim, num_filters)
        self.image_FPN = RefinePyramid(num_filters)
        self.cond_FPN = RefinePyramid(num_filters)
        self.aflow_net = AFlowNet(len(num_filters))
        self.old_lr = opt.lr
        self.old_lr_warp = opt.lr * 0.2

    def forward(self, cond_input, image_input, image_edge, cond):
        cond_pyramids = self.cond_FPN(self.cond_features(cond_input, cond))
        image_pyramids = self.image_FPN(self.image_features(image_input, cond))
        return self.aflow_net(image_input, image_edge, image_pyramids, cond_pyramids)

    def update_learning_rate(self, optimizer):
        lrd = opt.lr / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print(f'update learning rate: {self.old_lr} -> {lr}')
        self.old_lr = lr

    def update_learning_rate_warp(self, optimizer):
        lrd = 0.2 * opt.lr / opt.niter_decay
        lr = self.old_lr_warp - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print(f'update learning rate (warp): {self.old_lr_warp} -> {lr}')
        self.old_lr_warp = lr
