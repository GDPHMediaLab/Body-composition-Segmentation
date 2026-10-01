import argparse

import torch

from network import HybridUNet


def count_params(net):
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def show(name, obj):
    if obj is None:
        print(f"  {name:<16}: None")
    elif isinstance(obj, (list, tuple)):
        print(f"  {name:<16}: list, len={len(obj)}")
        for i, t in enumerate(obj):
            print(f"    [{i}] {tuple(t.shape)}")
    elif isinstance(obj, dict):
        print(f"  {name:<16}: dict")
        for k, v in obj.items():
            if isinstance(v, (list, tuple)):
                print(f"    {k:<16}: [{', '.join(str(tuple(t.shape)) for t in v)}]")
            elif v is None:
                print(f"    {k:<16}: None")
            else:
                print(f"    {k:<16}: {tuple(v.shape)}")
    else:
        print(f"  {name:<16}: {tuple(obj.shape)}")


def run(input_size, kwargs, tag):
    print(f"\n{'=' * 70}\nBuilding network: {tag}\n{'=' * 70}")
    net = HybridUNet(**kwargs)
    total, trainable, frozen = count_params(net)
    print(f"Parameters: total {total / 1e6:.2f}M | trainable {trainable / 1e6:.2f}M"
          f" | frozen ViT {frozen / 1e6:.2f}M")
    net.eval()

    x = torch.randn(1, 1, *input_size)

    print("\n--- forward (standard) ---")
    with torch.no_grad():
        seg = net.forward(x)
    show("seg(deep supervision logits)", seg)

    print("\n--- forward_detailed (full forward for teaching) ---")
    with torch.no_grad():
        out = net.forward_detailed(x)
    show("output dict", out)

    print("\n--- forward (with_vit=False, conv-only comparison) ---")
    kwargs2 = dict(kwargs)
    kwargs2['with_vit'] = False
    plain = HybridUNet(**kwargs2)
    plain.eval()
    with torch.no_grad():
        seg_plain = plain.forward(x)
    show("seg_plain", seg_plain)
    print("Trainable params of the conv-only branch:"
          f" {sum(p.numel() for p in plain.parameters() if p.requires_grad) / 1e6:.2f}M")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='quick check with a small config')
    parser.add_argument('--size', type=int, default=128,
                        help='input voxel side length for the full config (default 128, use 96 if memory is tight)')
    args = parser.parse_args()

    torch.manual_seed(0)

    if args.quick:
        run(
            input_size=(64, 64, 64),
            kwargs=dict(
                num_classes=3, in_channels=1,
                base_features=8, max_features=32, stages=5,
                vit_cube=64, vit_patch=16, vit_dim=96, vit_depth=2,
                vit_heads=4, vit_target_layers=(0, 1, 1)),
            tag='quick small config')
    else:
        run(
            input_size=(args.size,) * 3,
            kwargs=dict(
                num_classes=6, in_channels=1,
                base_features=32, stages=5, max_features=320,
                kernels=3, strides=(1, 2, 2, 2, 2),
                convs_per_stage=2, convs_per_stage_dec=None, conv_bias=True,
                deep_supervision=True,
                vit_cube=96, vit_patch=16, vit_dim=768, vit_depth=12,
                vit_heads=12, vit_target_layers=(3, 7, 11), freeze_vit=True,
                decoder_dropout=0.1, prior_hidden=256, with_vit=True),
            tag='v10 full config')
