from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform
from mmdet3d.models.fusion_models.deformAttention import DFA
from mmdet3d.models.fusion_models.bevfusion import DeformConv
__all__ = ["DepthLSSTransform"]


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()
        self.d_model_fv = 256 #80
        self.n_levels_fv = 1
        self.n_heads_fv = 8
        self.n_points_fv = 10
        self.fv_h, self.fv_w = feature_size
        # self.fv_h = 32
        # self.fv_w = 88
        self.num_att_fv = 1
        self.num_proj_fv = 1
        self.uv_deformAttn = True
        self.uv_deform = False
        self.deformAttn = DFA(self.d_model_fv, self.fv_h, self.fv_w, self.num_att_fv, self.num_proj_fv,
                              self.n_heads_fv, self.n_points_fv)
        self.deformconv = DeformConv(self.uv_deform)
        self.conv = nn.Conv2d(256, 80, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv = nn.Conv2d(80, 256, kernel_size=3, stride=1, padding=1, bias=True)

    @force_fp32()
    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        #print("before cat d shape:", d.shape)
        #print("before cat x shape:", x.shape)
        #x = self.conv(x)
        #print("after x shape:", x.shape)
        if self.uv_deformAttn:
            # x = self.conv(x)
            #print("after x shape:", x.shape)
            x = x + self.deformAttn(x)[0]
            # x = self.deconv(x)
            #print("after deformAttn x shape:", x.shape)
        if self.uv_deform:
            x = self.conv(x)
            x = x + self.deformconv(x)
            x = self.deconv(x)
        x = torch.cat([d, x], dim=1)
        #print("after cat x shape:", x.shape)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth

    def forward(self, *args, **kwargs):
        x, depth = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x, depth
