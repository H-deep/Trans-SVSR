# Trans-SVSR — Transformer-based Stereo Video Super-Resolution (x4)

**Trans-SVSR** is a transformer architecture for **stereo video super-resolution**.  
It reconstructs high-resolution frames by leveraging **left/right** views and **temporal** context.

> **Highlights**
> - Stereo + temporal transformer backbone
> - Patch-based training pipeline (x4)
> - Reproducible evaluation scripts
> - **Edge/production** export: ONNX → TensorRT (FP16/INT8) + benchmark harness

---

## Table of Contents
- [1. Environment](#1-environment)
- [2. Data Preparation](#2-data-preparation)
- [3. Training](#3-training)
- [4. Testing / Evaluation](#4-testing--evaluation)
- [5. Edge / Production Inference](#5-edge--production-inference)
- [6. Citation](#7-citation)

---

## 1. Environment

```bash
# conda (recommended)
conda create -n transsvsr python=3.10 -y
conda activate transsvsr
pip install -r requirements.txt
# (Optional) TensorRT / ONNXRuntime for edge export is described in Section 5
````

---

## 2. Data Preparation

1. **Download videos** for training (follow the dataset links/scripts you prefer).
2. Put the raw videos in:

```
data/raw_train/
    ├─ vid_0001.mp4
    ├─ vid_0002.mp4
    ├─ vid_0003.mp4
    └─ ...
```

3. **Create training patches (x4)**

```bash
python3 create_train_dataset.py
```

After this, patches are generated at:

```
data/train/patches_x4/
    ├─ sample_000001/
    │    ├─ l_000.png  l_001.png  l_002.png  l_003.png  l_004.png
    │    └─ r_000.png  r_001.png  r_002.png  r_003.png  r_004.png
    ├─ sample_000002/
    └─ ...
```

> Each **patch folder** contains **5 left** and **5 right** patches (temporal clip).
> Adjust clip length / stride in `create_train_dataset.py` if needed.

---

## 3. Training

```bash
python3 train.py --scale_factor 4 --device cuda:0 --batch_size 7 --lr 2e-3 --gamma 0.5 --start_epoch 0 --n_epochs 30 --n_steps 30 --trainset_dir ./data/train/ --model_name TransSVSR --load_pretrain False --model_path log/TransSVSR.pth.tar
```

---

## 4. Testing / Evaluation

First create the testing dataset.

###Creating the test set
Put the downloaded test videos in the dollowing path:
data/raw_test/

For SVSR-Set dataset, run the dollowing command:

python3 create_test_dataset_SVSRset.py

For NAMA3D and LFO3D datasets, run the dollowing command:

python3 create_test_dataset_nama_lfo.py

Please change the path accorfing to NAMA3D or LFO3D datasets. Nama3D [1] and LFO3D [2] need to be downloaded from their references and put in the /data/raw_test/ directory first.

```bash
# Single stereo sequence
python3 test.py \
  --model_name TransSVSR_4xSR \
  --testset_dir  ./data/test/ \

```
---

## 5. Edge / Production Inference

This repo includes a minimal **edge export + benchmark** path to demonstrate real-time deployment.

**Folder layout (added):**

```
edge/
  export_onnx.py          # PyTorch → ONNX
  build_trt.py            # ONNX → TensorRT (FP16/INT8)
  benchmark_infer.py      # latency/FPS/VRAM harness
  tiling.py               # optional high-res tiling utilities
  README_EDGE.md          # Jetson notes, power logging, tips
```

### 5.1 Export to ONNX

```bash
# Example input size for stereo (B=1, C=2, H=540, W=960). Adjust to your model.
python3 edge/export_onnx.py \
  --ckpt outputs/transsvsr_x4/best.pth \
  --onnx outputs/transsvsr_x4/model.onnx \
  --height 540 --width 960 --dynamic
```

### 5.2 Build TensorRT Engine (FP16 / INT8)

```bash
# FP16 engine
python3 edge/build_trt.py \
  --onnx outputs/transsvsr_x4/model.onnx \
  --engine outputs/transsvsr_x4/model_fp16.engine \
  --fp16 --workspace_gb 4

# INT8 (requires a small calibr. set; see README_EDGE.md)
python3 edge/build_trt.py \
  --onnx outputs/transsvsr_x4/model.onnx \
  --engine outputs/transsvsr_x4/model_int8.engine \
  --int8 --workspace_gb 4
```

### 5.3 Benchmark Latency / FPS / VRAM

```bash
python3 edge/benchmark_infer.py \
  --engine outputs/transsvsr_x4/model_fp16.engine \
  --height 540 --width 960 --batch 1
```

**Benchmark CSV schema (saved to `benchmarks/latency.csv`):**

```
device,input_h,input_w,precision,batch,fps,avg_latency_ms,vram_mb,power_w,notes,date
```

### 5.4 Jetson Notes (power & clocks)

```bash
# Set performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Power logging (1 Hz)
tegrastats --interval 1000 > tegrastats.log
```


## 6. Citation

If you use this repository, please cite our paper:

```bibtex
@inproceedings{imani2022new,
  title={A new dataset and transformer for stereoscopic video super-resolution},
  author={Imani, Hassan and Islam, Md Baharul and Wong, Lai-Kuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={706--715},
  year={2022}
}
```

---
