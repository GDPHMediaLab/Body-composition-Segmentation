import math

import torch
from torch import nn

from unet import _triple


def patch_grid(cube, patch):
    return tuple(i // p for i, p in zip(_triple(cube), _triple(patch)))


def sincos_positions_3d(grid, dim, extra_tokens=1):
    gd, gh, gw = grid
    total = gd * gh * gw
    pe = torch.zeros(total, dim)

    pos_d = torch.arange(gd).view(gd, 1, 1).expand(gd, gh, gw).reshape(-1)
    pos_h = torch.arange(gh).view(1, gh, 1).expand(gd, gh, gw).reshape(-1)
    pos_w = torch.arange(gw).view(1, 1, gw).expand(gd, gh, gw).reshape(-1)

    div = torch.exp(torch.arange(0, dim, 2).float()
                    * (-math.log(10000.0) / dim))
    for i, pos in enumerate((pos_d, pos_h, pos_w)):
        lo = i * dim // 3
        hi = dim if i == 2 else (i + 1) * dim // 3
        for j in range(lo, hi, 2):
            pe[:, j] = torch.sin(pos * div[(j - lo) // 2])
            if j + 1 < hi:
                pe[:, j + 1] = torch.cos(pos * div[(j - lo) // 2])

    return torch.cat([torch.zeros(extra_tokens, dim), pe], dim=0).unsqueeze(0)


class PathDrop(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = x.new_empty(shape).bernoulli_(keep)
        if keep > 0.0:
            noise.div_(keep)
        return x * noise


class CubeTokenizer(nn.Module):
    def __init__(self, cube=96, patch=16, cin=1, dim=768):
        super().__init__()
        self.cube, self.patch = _triple(cube), _triple(patch)
        self.grid = patch_grid(self.cube, self.patch)
        self.num_patches = self.grid[0] * self.grid[1] * self.grid[2]
        self.project = nn.Conv3d(cin, dim, kernel_size=self.patch,
                                 stride=self.patch)

    def forward(self, x):
        x = self.project(x)
        return x.flatten(2).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden=None, out=None, act=nn.GELU, drop=0.0):
        super().__init__()
        hidden = hidden or dim
        out = out or dim
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act()
        self.fc2 = nn.Linear(hidden, out)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TokenAttention(nn.Module):
    def __init__(self, dim, heads=12, qkv_bias=False, attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)) * self.scale
        a = self.attn_drop(a.softmax(dim=-1))
        x = (a @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_drop(self.out(x))


class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, qkv_bias=False,
                 attn_drop=0.0, proj_drop=0.0, path_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = TokenAttention(dim, heads, qkv_bias, attn_drop, proj_drop)
        self.path_drop = PathDrop(path_drop) if path_drop > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = FeedForward(dim, hidden=int(dim * mlp_ratio), drop=proj_drop)

    def forward(self, x):
        x = x + self.path_drop(self.attn(self.norm1(x)))
        x = x + self.path_drop(self.mlp(self.norm2(x)))
        return x


class CubeViT(nn.Module):
    def __init__(self, cube=96, patch=16, cin=1, dim=768, depth=12, heads=12,
                 mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop=0.0,
                 path_drop=0.0, use_learned_pos=False):
        super().__init__()
        self.cube = _triple(cube)
        self.patch = _triple(patch)
        self.tokenize = CubeTokenizer(cube, patch, cin, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        if use_learned_pos:
            self.pos = nn.Parameter(
                torch.zeros(1, 1 + self.tokenize.num_patches, dim))
        else:
            self.register_buffer(
                'pos', sincos_positions_3d(self.tokenize.grid, dim, 1))

        self.drop = nn.Dropout(drop_rate)
        drops = torch.linspace(0.0, path_drop, depth).tolist()
        self.layers = nn.ModuleList([
            TransformerLayer(dim, heads, mlp_ratio, qkv_bias,
                             attn_drop, drop_rate, drops[i])
            for i in range(depth)])
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.reset_weights()

    def reset_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.pos, nn.Parameter):
            nn.init.trunc_normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, keep_cls=False):
        x = self.tokenize(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.drop(x + self.pos)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x if keep_cls else x[:, 1:, :]
