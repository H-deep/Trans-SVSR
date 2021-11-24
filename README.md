# Trans-SVSR



The code for paper "A Transformer-based Model and a New Dataset for Stereoscopic Video Super-Resolution ".


# Overview #
Stereo video super-resolution (SVSR) aims to enhance the spatial resolution of the low-resolution video by reconstructing the high-resolution video. The key challenges in SVSR are preserving the stereo-consistency and temporal-consistency, without which viewers may experience 3D fatigue. There are several notable works on stereoscopic image super-resolution (SISR), but research on stereo video resolution is rare. In this paper, we propose a novel Transformer-based model for SVSR, namely \textit{Trans-SVSR}. \textit{Trans-SVSR} comprises two key novel components: a spatio-temporal convolutional self-attention layer and an optical flow-based feed-forward layer that discovers the correlation across different video frames and aligns the features. The parallax attention mechanism (PAM) that uses the cross-view information to consider the significant disparities is used for fusion of the stereo views. Due to the lack of the benchmark dataset suitable for the SVSR task, we collected a new stereoscopic video dataset, SVSR-Set.

# Requirements #

PyTorch 1.8.0+cu111, torchvision 0.9.0+cu111. The code is tested with python=3.6, cuda=10.2 with RTX 3090 GPU.


Run the following command for installing the requirements:

`pip3 install requirements`

# Training #

Follow the instructions bellow to first create the training set and then start training.

## Creating the tarining dataset ## 

1. Download the SVSR-Set dataset from the following link:

http://shorturl.at/mpwGX

2. Put the train set in the following path:

`data/raw_train/`

3. Start creating the train set:

`python3 create_train_dataset.py`


## Start training ## 

After creating the training dataset, start training with the following command:

`python3 train.py --scale_factor 4 --device cuda:0 --batch_size 7 --lr 2e-3 --gamma 0.5 --start_epoch 0 --n_epochs 30 --n_steps 30 --trainset_dir ./data/train/ --model_name TransSVSR --load_pretrain False --model_path log/TransSVSR.pth.tar`

The following parameters can be set:

```
--scale_factor
--device
--batch_size
--lr
--gamma
--start_epoch
--n_epochs
--n_steps
--trainset_dir
--model_name
--load_pretrain
--model_path
```

# Testing #
First create the testing dataset.

## Creating the test set ## 

1. Put the downloaded test videos in the dollowing path:

`data/raw_test/`

For SVSR-Set dataset, run the dollowing command:

`python3 create_test_dataset_SVSRset.py`


For NAMA3D and LFO3D datasets, run the dollowing command:

`python3 create_test_dataset_nama_lfo.py`

Please change the path accorfing to NAMA3D or LFO3D datasets. Nama3D [1] and LFO3D [2] need to be downloaded from their references and put in the /data/raw_test/ directory first.


## Start testing ## 

Run the following command:

`python3 test.py`

With the following parameters:

```
--testset_dir
--scale_factor
--device
--model_name
```

For testing the pre-trained Trans-SVSR model, first download the model from here: 
http://shorturl.at/dmtQR


Put the model in log/ folder, then run the test with default parameters on SVSR-Set dataset. For other datasets, specify the --testset_dir accorfingly.

We provided a video as the output of our method. The link is provided here:
http://shorturl.at/lsvP4

For creating the results for SISR-based methods that we compared them in our paper, we provided these method on the "SISR_methods" directory. This directory contains codes about the following methods:

```
PASSRNet [3]
iPASSR [4]
DFAM and its 4 versions [5]
SRRes+SAM [6]
```

For implemening these method, just follow the instructions inside each method provided by their owners.



# References #

[1] Matthieu Urvoy, Marcus Barkowsky, Romain Cousseau, Yao
Koudota, Vincent Ricorde, Patrick Le Callet, Jesus Gutierrez,
and Narciso Garcia. Nama3ds1-cospad1: Subjective video
quality assessment database on coding conditions introducing
freely available high quality 3d stereoscopic sequences. In
Fourth International Workshop on Quality of Multimedia Experience, pages 109–114. IEEE, 2012. 1, 2, 3

[2] Balasubramanyam Appina, Sathya Veera Reddy Dendi, K
Manasa, Sumohana S Channappayya, and Alan C Bovik.
Study of subjective quality and objective blind quality prediction of stereoscopic videos. IEEE Transactions on Image
Processing, 28(10):5027–5040, 2019. 1, 2, 3

[3] Longguang Wang, Yingqian Wang, Zhengfa Liang, Zaiping
Lin, Jungang Yang, Wei An, and Yulan Guo. Learning parallax attention for stereo image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 12250–12259, 2019. 1, 2, 5, 7, 8

[4] Yingqian Wang, Xinyi Ying, Longguang Wang, Jungang
Yang, Wei An, and Yulan Guo. Symmetric parallax attention for stereo image super-resolution. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 766–775, 2021. 1, 2, 7, 8

[5] Jiawang Dan, Zhaowei Qu, Xiaoru Wang, and Jiahang Gu. A
disparity feature alignment module for stereo image superresolution. IEEE Signal Processing Letters, 2021. 7, 8

[6] Xinyi Ying, Yingqian Wang, Longguang Wang, Weidong
Sheng, Wei An, and Yulan Guo. A stereo attention module
for stereo image super-resolution. IEEE Signal Processing
Letters, 27:496–500, 2020. 1, 7
