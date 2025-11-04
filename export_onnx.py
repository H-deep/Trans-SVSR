#!/usr/bin/env python3

import os
import sys
import types
import argparse
import torch
import torch.nn as nn

fake_args = types.SimpleNamespace()
fake_args.scale_factor = 4
fake_args.device = "cuda"

vespcn_pkg = types.ModuleType("VESPCN")
option_mod = types.ModuleType("VESPCN.option")
setattr(option_mod, "args", fake_args)

sys.modules["VESPCN"] = vespcn_pkg
sys.modules["VESPCN.option"] = option_mod

from model_simple import Net  # Net(upscale_factor, spatial_dim, cfg)


def cli():
    ap = argparse.ArgumentParser("Export Trans-SVSR to ONNX")
    ap.add_argument("--ckpt", required=True, help="Path to .pth or .pth.tar")
    ap.add_argument("--onnx", required=True, help="Output ONNX path")
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--frames", type=int, default=5, help="temporal length T")
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--simplify", action="store_true")
    return ap.parse_args()


def disable_bn_running_stats(module: nn.Module):
    # Mirror your test-time BN handling (typoed names kept intentionally)
    for m in module.modules():
        for child in m.children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(child, "track_runing_stats", False)
                setattr(child, "runing_mean", None)
                setattr(child, "runing_var", None)
            if isinstance(child, nn.BatchNorm3d):
                setattr(child, "track_runing_stats", False)
                setattr(child, "runing_mean", None)
                setattr(child, "runing_var", None)


class NetWrapper(nn.Module):
    """Expose only tensor inputs; force is_training=0 for ONNX."""
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        out_l, out_r = self.net(x_left, x_right, is_training=0)
        return out_l, out_r


def main():
    a = cli()
    os.makedirs(os.path.dirname(a.onnx), exist_ok=True)

    # Build a minimal cfg for Net (replacing VESPCN.option.args at runtime)
    cfg = types.SimpleNamespace()
    cfg.scale_factor = a.scale
    cfg.device = a.device

    spatial_dim = (a.height, a.width)

    # Build and load model
    device = torch.device(a.device)
    net = Net(upscale_factor=cfg.scale_factor, spatial_dim=spatial_dim, cfg=cfg).to(device)

    ckpt = torch.load(a.ckpt, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[export] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:    print("  missing:", missing[:12], "..." if len(missing) > 12 else "")
        if unexpected: print("  unexpected:", unexpected[:12], "..." if len(unexpected) > 12 else "")

    disable_bn_running_stats(net)
    net.eval()
    torch.set_grad_enabled(False)

    wrapped = NetWrapper(net).to(device).eval()

    # Dummy inputs (B,C,T,H,W)
    B, C, T, H, W = 1, a.channels, a.frames, a.height, a.width
    x_left  = torch.randn(B, C, T, H, W, device=device)
    x_right = torch.randn(B, C, T, H, W, device=device)

    dynamic_axes = None
    if a.dynamic:
        dynamic_axes = {
            "x_left":    {0: "batch", 2: "time", 3: "height", 4: "width"},
            "x_right":   {0: "batch", 2: "time", 3: "height", 4: "width"},
            "out_left":  {0: "batch", 2: "height_up", 3: "width_up"},
            "out_right": {0: "batch", 2: "height_up", 3: "width_up"},
        }

    print(f"[export] Exporting ONNX → {a.onnx} (opset={a.opset}, dynamic={bool(a.dynamic)})")
    torch.onnx.export(
        wrapped,
        (x_left, x_right),
        a.onnx,
        input_names=["x_left", "x_right"],
        output_names=["out_left", "out_right"],
        opset_version=a.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    print("[export] ONNX saved.")

    if a.simplify:
        try:
            import onnx
            from onnxsim import simplify
            print("[export] simplifying ...")
            m = onnx.load(a.onnx)
            ms, check = simplify(m)
            assert check, "onnx-simplifier check failed"
            onnx.save(ms, a.onnx)
            print("[export] simplified ONNX saved.")
        except Exception as e:
            print(f"[export] simplifier skipped: {e}")

    print("[export] Done.")


if __name__ == "__main__":
    main()
