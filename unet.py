import torch
from torch import nn


def _triple(v):
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v, v, v)


class ConvUnit(nn.Module):
    def __init__(self, cin, cout, kernel, stride, bias=False,
                 norm=nn.InstanceNorm3d, act=nn.LeakyReLU, dropout=None):
        super().__init__()
        k, s = _triple(kernel), _triple(stride)
        ops = [nn.Conv3d(cin, cout, k, s,
                         padding=[(i - 1) // 2 for i in k], bias=bias)]
        if dropout is not None:
            ops.append(dropout())
        if norm is not None:
            ops.append(norm(cout))
        if act is not None:
            ops.append(act(inplace=True))
        self.path = nn.Sequential(*ops)

    def forward(self, x):
        return self.path(x)


class ConvSequence(nn.Module):
    def __init__(self, num_units, cin, cout, kernel, first_stride, **unit_kw):
        super().__init__()
        units = [ConvUnit(cin, cout, kernel, first_stride, **unit_kw)]
        for _ in range(num_units - 1):
            units.append(ConvUnit(cout, cout, kernel, 1, **unit_kw))
        self.units = nn.Sequential(*units)

    def forward(self, x):
        return self.units(x)


class DownPath(nn.Module):
    def __init__(self, cin, n_stages, channels, kernels, strides, num_units,
                 **unit_kw):
        super().__init__()
        channels = list(channels)
        self.out_channels = channels
        self.strides = [_triple(s) for s in strides]
        self.kernels = list(kernels)

        stages = []
        c = cin
        for i in range(n_stages):
            stages.append(ConvSequence(num_units[i], c, channels[i],
                                       kernels[i], strides[i], **unit_kw))
            c = channels[i]
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class UpPath(nn.Module):
    def __init__(self, down_path, num_classes, num_units, deep_supervision=True,
                 **unit_kw):
        super().__init__()
        self.deep_supervision = deep_supervision
        n_stages = len(down_path.out_channels)

        ups, stages, segs = [], [], []
        for s in range(1, n_stages):
            ch_below = down_path.out_channels[-s]
            ch_skip = down_path.out_channels[-(s + 1)]
            ups.append(nn.ConvTranspose3d(
                ch_below, ch_skip,
                kernel_size=down_path.strides[-s],
                stride=down_path.strides[-s],
                bias=unit_kw.get('bias', False)))
            stages.append(ConvSequence(num_units[s - 1], ch_skip * 2, ch_skip,
                                       down_path.kernels[-(s + 1)], 1, **unit_kw))
            segs.append(nn.Conv3d(ch_skip, num_classes, 1, bias=True))

        self.ups = nn.ModuleList(ups)
        self.stages = nn.ModuleList(stages)
        self.segs = nn.ModuleList(segs)

    def forward(self, feats):
        x = feats[-1]
        outs = []
        for s in range(len(self.stages)):
            x = self.ups[s](x)
            x = torch.cat([x, feats[-(s + 2)]], dim=1)
            x = self.stages[s](x)
            if self.deep_supervision:
                outs.append(self.segs[s](x))
            elif s == len(self.stages) - 1:
                outs.append(self.segs[-1](x))
        outs = outs[::-1]
        return outs if self.deep_supervision else outs[0]


class BasicUNet(nn.Module):
    def __init__(self, cin, n_stages, channels, kernels, strides, num_units,
                 num_classes, num_units_dec=None, deep_supervision=True,
                 bias=True, norm=nn.InstanceNorm3d, act=nn.LeakyReLU,
                 dropout=None):
        super().__init__()
        num_units = [num_units] * n_stages if isinstance(num_units, int) \
            else list(num_units)
        if isinstance(kernels, int):
            kernels = [kernels] * n_stages
        else:
            kernels = list(kernels)
        if isinstance(strides, int):
            strides = [strides] * n_stages
        else:
            strides = list(strides)
        if num_units_dec is None:
            num_units_dec = [2] * (n_stages - 1)
        elif isinstance(num_units_dec, int):
            num_units_dec = [num_units_dec] * (n_stages - 1)

        unit_kw = dict(bias=bias, norm=norm, act=act, dropout=dropout)
        self.encoder = DownPath(cin, n_stages, channels, kernels, strides,
                                num_units, **unit_kw)
        self.decoder = UpPath(self.encoder, num_classes, num_units_dec,
                              deep_supervision, **unit_kw)

    def forward(self, x):
        return self.decoder(self.encoder(x))
