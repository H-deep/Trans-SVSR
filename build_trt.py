#!/usr/bin/env python3
"""
Build a TensorRT engine for Trans-SVSR (two inputs, two outputs, 5D tensors).

Works across TRT versions:
- TRT >= 8.2+: uses builder.build_serialized_network(...) → deserialize
- Older: falls back to builder.build_engine(...) if present

Examples
--------
# FP16 (fixed 5x240x240)
python3 edge/build_trt.py \
  --onnx outputs/transsvsr_x4/model_static.onnx \
  --engine outputs/transsvsr_x4/model_fp16.engine \
  --fp16 \
  --min_T 5 --opt_T 5 --max_T 5 \
  --min_H 240 --opt_H 240 --max_H 240 \
  --min_W 240 --opt_W 240 --max_W 240

# INT8 with calibration pairs (left_*.npy/right_*.npy shaped (1,3,T,H,W))
python3 edge/build_trt.py \
  --onnx outputs/transsvsr_x4/model_static.onnx \
  --engine outputs/transsvsr_x4/model_int8.engine \
  --int8 --calib_dir calib_samples/ \
  --opt_T 5 --opt_H 540 --opt_W 960
"""

import os
import glob
import argparse
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _set_workspace(config, gb: float = 4.0):
    bytes_ = int(gb) << 30
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_)
    except Exception:
        # very old TRT
        config.max_workspace_size = bytes_


# ----------------------------- INT8 Calibrator ----------------------------- #

class NumpyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Simple calibrator for two-input network (x_left, x_right).
    Provide --calib_dir with left_*.npy and right_*.npy batches shaped
    (B, C, T, H, W) or (C, T, H, W) (batch dim will be added).
    """
    def __init__(self, input_shapes, cache_file="calib.cache", calib_dir=None, max_batches=20):
        super().__init__()
        self.cache_file = cache_file
        self.max_batches = max_batches
        self.batch_idx = 0
        self.shapes = input_shapes

        self.left_files = []
        self.right_files = []
        if calib_dir and os.path.isdir(calib_dir):
            lf = sorted(glob.glob(os.path.join(calib_dir, "left_*.npy")))
            rf = sorted(glob.glob(os.path.join(calib_dir, "right_*.npy")))
            n = min(len(lf), len(rf))
            self.left_files = lf[:n]
            self.right_files = rf[:n]
            self.max_batches = min(self.max_batches, n)

        # Device buffers
        self.device_buffers = []
        for s in self.shapes:
            vol = int(np.prod(s))
            self.device_buffers.append(trt.cuda_malloc(vol * 4))  # float32

    def get_batch_size(self):
        return self.shapes[0][0]

    def get_batch(self, names):
        if self.batch_idx >= self.max_batches:
            return None

        if self.left_files:
            left = np.load(self.left_files[self.batch_idx])
            right = np.load(self.right_files[self.batch_idx])
        else:
            # fallback: random data just to build an engine
            left = np.random.rand(*self.shapes[0]).astype(np.float32)
            right = np.random.rand(*self.shapes[1]).astype(np.float32)

        if left.ndim == 4:
            left = np.expand_dims(left, 0)
        if right.ndim == 4:
            right = np.expand_dims(right, 0)

        trt.memcpy_htod(self.device_buffers[0], left)
        trt.memcpy_htod(self.device_buffers[1], right)

        self.batch_idx += 1
        return [int(self.device_buffers[0]), int(self.device_buffers[1])]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# ----------------------------- Engine Builder ------------------------------ #

def build_engine(
    onnx_path: str,
    engine_path: str,
    min_shape=(1, 3, 1, 256, 256),
    opt_shape=(1, 3, 5, 540, 960),
    max_shape=(1, 3, 5, 1080, 1920),
    fp16=False,
    int8=False,
    calib_dir=None,
    workspace_gb=4.0,
):
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(onnx_path)

    builder = trt.Builder(TRT_LOGGER)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX parse error:", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    _set_workspace(config, workspace_gb)

    # Optimization profile for both inputs (assumed order: x_left, x_right)
    profile = builder.create_optimization_profile()

    in0 = network.get_input(0)
    in1 = network.get_input(1)
    name0 = in0.name
    name1 = in1.name

    def _set_profile_for_input(name):
        profile.set_shape(name, min_shape, opt_shape, max_shape)

    _set_profile_for_input(name0)
    _set_profile_for_input(name1)
    config.add_optimization_profile(profile)

    # Precision flags
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        if not builder.platform_has_fast_int8:
            print("Warning: Platform does not report fast INT8; continuing anyway.")
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = NumpyEntropyCalibrator(
            input_shapes=[opt_shape, opt_shape],
            cache_file=os.path.splitext(engine_path)[0] + ".calib",
            calib_dir=calib_dir,
            max_batches=20,
        )
        config.int8_calibrator = calibrator

    print(f"[TRT] Building engine → {engine_path}")
    engine = None

    # Preferred path: TRT >= 8.2
    if hasattr(builder, "build_serialized_network"):
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("build_serialized_network() returned None")
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(plan)
    # Older path: some versions expose build_engine(network, config)
    elif hasattr(builder, "build_engine"):
        engine = builder.build_engine(network, config)
    # Very old path (no config): build_cuda_engine(network)
    elif hasattr(builder, "build_cuda_engine"):
        print("[TRT] Warning: Falling back to build_cuda_engine(network) (very old API). "
              "Precision/profile controls may be limited.")
        engine = builder.build_cuda_engine(network)
    else:
        raise AttributeError("No suitable TensorRT builder method found on this installation.")

    if engine is None:
        raise RuntimeError("Engine build failed.")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"[TRT] Saved: {engine_path}")

    # I/O summary
    try:
        print("[TRT] IO tensors:")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            shape = engine.get_tensor_shape(name)
            print(f"  - {name} ({'IN' if mode==trt.TensorIOMode.INPUT else 'OUT'}): {shape}")
    except Exception:
        pass


def parse_args():
    ap = argparse.ArgumentParser("Build TensorRT engine for Trans-SVSR")
    ap.add_argument("--onnx", required=True, help="Path to ONNX file")
    ap.add_argument("--engine", required=True, help="Output .engine path")
    ap.add_argument("--workspace_gb", type=float, default=4.0)

    # Precision
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib_dir", type=str, default=None, help="Folder with left_*.npy/right_*.npy")

    # Shape profile (B,C,T,H,W) — keep B=1, C=3 normally
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--min_T", type=int, default=1)
    ap.add_argument("--opt_T", type=int, default=5)
    ap.add_argument("--max_T", type=int, default=5)
    ap.add_argument("--min_H", type=int, default=256)
    ap.add_argument("--opt_H", type=int, default=540)
    ap.add_argument("--max_H", type=int, default=1080)
    ap.add_argument("--min_W", type=int, default=256)
    ap.add_argument("--opt_W", type=int, default=960)
    ap.add_argument("--max_W", type=int, default=1920)
    return ap.parse_args()


def main():
    a = parse_args()
    B = a.batch
    C = a.channels

    min_shape = (B, C, a.min_T, a.min_H, a.min_W)
    opt_shape = (B, C, a.opt_T, a.opt_H, a.opt_W)
    max_shape = (B, C, a.max_T, a.max_H, a.max_W)

    build_engine(
        onnx_path=a.onnx,
        engine_path=a.engine,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        fp16=a.fp16,
        int8=a.int8,
        calib_dir=a.calib_dir,
        workspace_gb=a.workspace_gb,
    )


if __name__ == "__main__":
    main()
