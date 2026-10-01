import torch
import torch.nn as nn
import torch.nn.functional as F


class DimBridge(nn.Module):
    def __init__(self, vit_dim, target, mid=None):
        super().__init__()
        if mid is None:
            mid = max(target * 2, vit_dim // 2)
        self.path = nn.Sequential(
            nn.Conv3d(vit_dim, mid, kernel_size=1, bias=False),
            nn.InstanceNorm3d(mid),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid, target, kernel_size=1, bias=False),
            nn.InstanceNorm3d(target),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.path(x)


class GateFusion(nn.Module):
    def __init__(self, target, vit_dim=768):
        super().__init__()
        self.bridge = DimBridge(vit_dim, target)

        self.refine = nn.Sequential(
            nn.Conv3d(target, target, kernel_size=1, bias=False),
            nn.InstanceNorm3d(target),
            nn.LeakyReLU(inplace=True),
        )

        def _gate_seq():
            return nn.Sequential(
                nn.Conv3d(target, target, 1, bias=False),
                nn.InstanceNorm3d(target),
                nn.ReLU(inplace=True))

        self.conv_gate = _gate_seq()
        self.vit_gate = _gate_seq()
        self.gate_head = nn.Sequential(
            nn.Conv3d(target, target, 1, bias=False),
            nn.InstanceNorm3d(target),
            nn.Sigmoid())

        self.spatial_gate = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=5, padding=2, bias=False),
            nn.Conv3d(1, 1, kernel_size=1, bias=False),
            nn.Sigmoid())

        self.merge = nn.Conv3d(target * 2, target, kernel_size=1, bias=False)

    def forward(self, conv_feat, vit_feat):
        identity = conv_feat

        if vit_feat is None:
            vit_feat = conv_feat.new_zeros(
                conv_feat.shape[0], self.bridge.path[0].in_channels,
                *conv_feat.shape[2:])

        v = self.bridge(vit_feat)
        if v.shape[2:] != conv_feat.shape[2:]:
            v = F.interpolate(v, size=conv_feat.shape[2:],
                              mode='trilinear', align_corners=False)
            v = self.refine(v)

        gate = self.gate_head(self.conv_gate(conv_feat) + self.vit_gate(v))
        fused = torch.cat([gate * conv_feat, gate * v], dim=1)

        stat = torch.cat([fused.mean(dim=1, keepdim=True),
                          fused.amax(dim=1, keepdim=True)], dim=1)
        out = self.spatial_gate(stat) * fused
        out = self.merge(out)

        return out + identity
