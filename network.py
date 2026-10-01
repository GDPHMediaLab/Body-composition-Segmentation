import torch
import torch.nn as nn
import torch.nn.functional as F

from unet import BasicUNet
from transformer import CubeViT
from FSMG import GateFusion


class FeatureCollector(nn.Module):
    def __init__(self, transformer, target_layers=(3, 7, 11), freeze=True):
        super().__init__()
        self.transformer = transformer
        self.targets = tuple(target_layers)
        self.freeze = freeze
        if freeze:
            for p in transformer.parameters():
                p.requires_grad = False
        self.saved = {}
        for idx in self.targets:
            transformer.layers[idx].register_forward_hook(self._make_hook(idx))

    def _make_hook(self, idx):
        def hook(module, args, out):
            self.saved[idx] = out
        return hook

    def forward(self, x):
        with torch.set_grad_enabled(not self.freeze):
            self.saved.clear()
            self.transformer(x)
        grid = self.transformer.tokenize.grid
        out = {}
        for idx, tokens in self.saved.items():
            spatial = tokens[:, 1:, :].transpose(1, 2)
            out[idx] = spatial.reshape(tokens.shape[0], -1, *grid).contiguous()
        return out


class LogitUpsampler(nn.Module):
    def __init__(self, num_classes, times):
        super().__init__()
        layers = []
        for _ in range(times):
            layers += [
                nn.ConvTranspose3d(num_classes, num_classes,
                                   kernel_size=2, stride=2),
                nn.InstanceNorm3d(num_classes),
                nn.LeakyReLU(0.01, inplace=True),
            ]
        self.upscale = nn.Sequential(*layers)

    def forward(self, x):
        return self.upscale(x)


class HybridUNet(nn.Module):
    def __init__(self, num_classes, in_channels=1,
                 base_features=32, stages=5, max_features=320,
                 kernels=3, strides=(1, 2, 2, 2, 2),
                 convs_per_stage=2, convs_per_stage_dec=None, conv_bias=True,
                 deep_supervision=True,
                 vit_cube=96, vit_patch=16, vit_dim=768, vit_depth=12,
                 vit_heads=12, vit_target_layers=(3, 7, 11), freeze_vit=True,
                 decoder_dropout=0.1, prior_hidden=256, with_vit=True):
        super().__init__()
        self.with_vit = with_vit
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.decoder_dropout = decoder_dropout
        self.vit_cube = vit_cube
        self.vit_targets = tuple(vit_target_layers)

        channels = [min(base_features * 2 ** i, max_features)
                    for i in range(stages)]
        self.channels = channels

        self.base = BasicUNet(
            in_channels, stages, channels, kernels, strides, convs_per_stage,
            num_classes, convs_per_stage_dec, deep_supervision, conv_bias,
            nn.InstanceNorm3d, nn.LeakyReLU, None)

        if with_vit:
            self.collector = FeatureCollector(
                CubeViT(cube=vit_cube, patch=vit_patch, cin=in_channels,
                        dim=vit_dim, depth=vit_depth, heads=vit_heads),
                target_layers=self.vit_targets, freeze=freeze_vit)
            self.fuse2 = GateFusion(channels[2], vit_dim)
            self.fuse3 = GateFusion(channels[3], vit_dim)
            self.fuse4 = GateFusion(channels[4], vit_dim)
            self.prior_head = nn.Sequential(
                nn.Conv3d(vit_dim, prior_hidden, 1),
                nn.InstanceNorm3d(prior_hidden),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(prior_hidden, num_classes, 1))
            self.proj_n = nn.Conv3d(channels[3], num_classes, 1)
            self.logit_ups = nn.ModuleDict({
                's1': LogitUpsampler(num_classes, 1),
                's2': LogitUpsampler(num_classes, 2),
                's3': LogitUpsampler(num_classes, 3),
            })

    def encode_with_fusion(self, x, vit_maps):
        feats = []
        for i, stage in enumerate(self.base.encoder.stages):
            x = stage(x)
            if self.with_vit:
                if i == 2:
                    x = self.fuse2(x, vit_maps.get(self.vit_targets[0]))
                elif i == 3:
                    x = self.fuse3(x, vit_maps.get(self.vit_targets[1]))
                elif i == 4:
                    x = self.fuse4(x, vit_maps.get(self.vit_targets[2]))
            feats.append(x)
        return feats

    def forward(self, x):
        vit_maps = None
        if self.with_vit:
            cube = F.interpolate(x, size=(self.vit_cube,) * 3,
                                 mode='trilinear', align_corners=False)
            vit_maps = self.collector(cube)
        feats = self.encode_with_fusion(x, vit_maps)
        return self.base.decoder(feats)

    def forward_detailed(self, x):
        vit_maps = None
        if self.with_vit:
            cube = F.interpolate(x, size=(self.vit_cube,) * 3,
                                 mode='trilinear', align_corners=False)
            vit_maps = self.collector(cube)
        feats = self.encode_with_fusion(x, vit_maps)
        bottleneck = feats[-1]

        if self.training and self.decoder_dropout > 0:
            feats_in = [F.dropout3d(f, p=self.decoder_dropout) if i >= 2 else f
                        for i, f in enumerate(feats)]
        else:
            feats_in = feats

        seg = self.base.decoder(feats_in)
        if not isinstance(seg, (list, tuple)):
            seg = [seg]

        seg_up = prior_logits = prior_logits_up = feat_n = last_vit = None
        if self.with_vit:
            seg_up = [seg[0],
                      self.logit_ups['s1'](seg[1]),
                      self.logit_ups['s2'](seg[2]),
                      self.logit_ups['s3'](seg[3])]
            last_vit = vit_maps[self.vit_targets[2]]
            prior_logits = F.interpolate(
                self.prior_head(last_vit), size=(32,) * 3,
                mode='trilinear', align_corners=False)
            prior_logits_up = self.logit_ups['s2'](prior_logits)
            feat_n = self.proj_n(feats[3])

        return dict(
            seg=seg,
            seg_up=seg_up,
            bottleneck=bottleneck,
            skip1=feats[1],
            skip3=feats[3],
            vit_last=last_vit,
            prior_logits=prior_logits,
            prior_logits_up=prior_logits_up,
            feat_n=feat_n,
        )


def load_mae_weights(net, ckpt_path):
    vit = net.collector.transformer
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model') or ckpt.get('state_dict') or ckpt

    remap = {}
    for k, v in state.items():
        k2 = k
        if k2 == 'encoder_pos_embed':
            remap['pos'] = v
            continue
        if k2.startswith('encoder.'):
            k2 = k2[len('encoder.'):]
        if k2 == 'pos_embed':
            k2 = 'pos'
        if k2.startswith('patch_embed.'):
            k2 = 'tokenize.' + k2[len('patch_embed.'):]
        elif k2.startswith('blocks.'):
            k2 = 'layers.' + k2[len('blocks.'):]
        elif k2 == 'norm.weight':
            k2 = 'final_norm.weight'
        elif k2 == 'norm.bias':
            k2 = 'final_norm.bias'
        k2 = k2.replace('.attn.proj.', '.attn.out.')
        remap[k2] = v

    if 'pos' in remap and remap['pos'].ndim == 3 and \
            remap['pos'].shape[1] != vit.pos.shape[1]:
        pad = torch.zeros(remap['pos'].shape[0], 1, remap['pos'].shape[2])
        remap['pos'] = torch.cat([pad, remap['pos']], dim=1)
    for k in list(remap):
        target = vit.state_dict().get(k)
        if target is not None and target.shape != remap[k].shape:
            del remap[k]

    result = vit.load_state_dict(remap, strict=False)
    return result
