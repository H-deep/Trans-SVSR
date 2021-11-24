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

http://shorturl.at/mpw

2. Put the train set in the following path:

`data/raw_train/`

3. Start creating the train set:

`python3 create_train_dataset.py`


## Start training ## 

After creating the training dataset, start training with the following command:

`python3 train.py `

The following parameters can be set:

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


# Testing #
First create the resting dataset.

## Creating the test set ## 

1. Put the test set in the dollowing path:

`data/raw_test/`

For SVSR-Set dataset, run the dollowing command:

`python3 create_test_dataset_SVSRset.py`


For NAMA3D and LFO3D datasets, run the dollowing command:

`python3 create_test_dataset_nama_lfo.py`

Please change the path accorfing to NAMA3D or LFO3D datasets.


## Start testing ## 

Run the following command:

`python3 test.py`

With the following parameters:

--testset_dir
--scale_factor
--device
--model_name
    

For testing the pre-trained Trans-SVSR model, first download the model from here: 
https://drive.google.com/file/d/1b3cwOBRfVRO0rK13rsPt71zp8N-Dxb24/view?usp=sharing

Put the model in log/ folder, then run the test with default parameters on SVSR-Set dataset. For other datasets, specify the --testset_dir accorfingly.
