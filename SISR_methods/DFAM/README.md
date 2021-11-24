# A Disparity Feature Alignment Module for Stereo Image Super-Resolution

### Abstract
Recently, the performance of super-resolution has been improved by the stereo images since the additional information could be obtained from another view. However, it is a challenge to interact the cross-view information since disparities between left and right images are variable. To address this issue, we propose a disparity feature alignment module (DFAM) to exploit the disparity information for feature alignment and fusion. Specifically, we design a modified atrous spatial pyramid pooling module to estimate disparities and warp stereo features. Then we use spatial and channel attention for feature fusion. In addition, DFAM can be plugged into an arbitrary SISR network to super-resolve a stereo image pair. Extensive experiments demonstrate that DFAM incorporates stereo information with less inference time and memory cost. Moreover, RCAN equipped with DFAMs achieves better performance against state-of-the-art methods. The code can be obtained at https://github.com/JiawangDan/DFAM.

### Requirements
- Python 3
- PyTorch, torchvision
- Numpy, Scipy
- importlib
- Matlab

### Dataset
We also use 800 images and 112 images of Flickr1024 dataset as the training data and the validation data respectively. In addition, we use 5 images from the Middlebury dataset, 20 images from the KITTI 2012 dataset and 20 images from the KITTI 2015 dataset as the test data.

1. Download [Flickr1024](https://yingqianwang.github.io/Flickr1024) and unzip on `dataset` directory as below:
  ```
  data
  └── train
      ├── Flickr1024
          ├── 0001_L.png
          ├── 0001_R.png
          ├── 0002_L.png
          ├── ...
      ├── Flickr1024_patches
          ├── patches_x2
          ├── patches_x3
          ├── patches_x4
      ├── generate_trainset.m
      ├── modcrop.m
  └── valid
      ├── ...
  └── test
      ├── middlebury
          ├── hr
              ├── cloth2
                  ├── lr0.png
                  ├── lr1.png
              ├── ...
          ├── lr_x2
              ├── ...
          ├── lr_x4
              ├── ...
      ├── KITTI2012
          ├── ...
      ├── KITTI2015
          ├── ...
  ```
2. During the training process, all the training data is cropped into 30×90 patches with a stride of 20. Move 'generate_testset.m', 'generate_trainset.m' and 'modcrop.m' in the above location.
```shell
$ cd data/train && python generate_trainset.py
$ cd data/test && python generate_testset.py
```
3. Other benchmark datasets can be downloaded in [Middlebury](https://vision.middlebury.edu/stereo/), [KITTI2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) and [KITTI2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo). Please put all the datasets in `data` directory.

### Test Pretrained Models
We provide the pretrained models in [ckpt](https://pan.baidu.com/s/1gwBtig-SIOzrpEyMczLfkw) (提取码：1234) directory. To test DFAM on benchmark dataset:
```shell
$ python test.py --model VDSR_DFAM --scale 4 --dataset middlebury --upsample --rgb2y --checkpoint ckpt/VDSR_DFAM/VDSR_DFAM_x4.pth --device cuda
$ python test.py --model SRCNN_DFAM --scale 4 --dataset middlebury --upsample --rgb2y --checkpoint ckpt/SRCNN_DFAM/SRCNN_DFAM_x4.pth --device cuda
$ python test.py --model SRResNet_DFAM --scale 4 --dataset middlebury --checkpoint ckpt/SRResNet_DFAM/SRResNet_DFAM_x4.pth --device cuda
$ python test.py --model RCAN_DFAM --scale 4 --dataset middlebury --checkpoint ckpt/RCAN_DFAM/RCAN_DFAM_x4.pth --device cuda
```

### Training Models
To augment the data, random horizontal and vertical flipping are adopted. 
```shell
$ python train.py --model VDSR_DFAM --scale 2 --batchSize 12 --upsample --rgb2y --pretrained ./ckpt/VDSR/pretrain_statedict.pth
$ python train.py --model SRCNN_DFAM --scale 2 --batchSize 16 --upsample --rgb2y --pretrained ./ckpt/SRCNN/pretrain_statedict.pth
$ python train.py --model SRResNet_PDAM --scale 2 --batchSize 32 --pretrained ./ckpt/SRResNet/pretrain_statedictx2.pth
$ python train.py --model RCAN_PDAM --scale 2 --batchSize 8 --pretrained ./ckpt/RCAN/pretrain_statedictx2.pt
```
### Results
![image](https://github.com/JiawangDan/DFAM/blob/main/fig/results1.png)

### Citation
```
@article{
  title={A Disparity Feature Alignment Module for Stereo Image Super-Resolution},
  author={Jiawang Dan, Zhaowei Qu, Xiaoru Wang, and Jiahang Gu},
  year={2021}
}
```
