import cv2
import os
import numpy as np
from PIL import Image
import torch 

import random

scale = 4
data_dir = "data/train/patches_x4/"
data_npz_dir = "data/train/patches_x4_npz/"
list_patches = os.listdir(data_dir)
ret2 = True
patch_w = 32
patch_h = 88
patch_w_hr = 32*scale
patch_h_hr = 88*scale
patch_counter = 0
patches_4_lr_left = torch.empty(4, 3,5,32,88)
patches_4_lr_right = torch.empty(4, 3,5,32,88)
patches_4_hr_left = torch.empty(4, 3,32*4,88*4)
patches_4_hr_right = torch.empty(4, 3,32*4,88*4)

npz_counter = 0

random.shuffle(list_patches)

def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)

for patch in list_patches:
    patch_counter += 1
    
    



    img_lr_left_list = torch.empty(3,5,32,88)
    img_lr_right_list = torch.empty(3,5,32,88)
    for i in range(1,6):
        img_hr_left  = Image.open(data_dir + '/' + patch + '/'+str(3)+ '/hr0.png')
        img_hr_right = Image.open(data_dir + '/' + patch + '/'+str(3)+ '/hr1.png')
        img_lr_left  = Image.open(data_dir + '/' + patch + '/'+str(i)+ '/lr0.png')
        img_lr_right = Image.open(data_dir + '/' + patch + '/'+str(i)+ '/lr1.png')
        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        img_hr_left_list = toTensor(img_hr_left)
        img_hr_right_list = toTensor(img_hr_right)
        img_lr_left_list[:,i-1,:,:] = toTensor(img_lr_left)
        img_lr_right_list[:,i-1,:,:] = toTensor(img_lr_right)



    patches_4_lr_left[patch_counter-1] = img_lr_left_list
    patches_4_lr_right[patch_counter-1] = img_lr_right_list
    patches_4_hr_left[patch_counter-1] = img_hr_left_list
    patches_4_hr_right[patch_counter-1] = img_hr_right_list


    if(patch_counter==4):
        patches_4_lr_left = patches_4_lr_left.cpu().detach().numpy()
        patches_4_lr_right = patches_4_lr_right.cpu().detach().numpy()
        patches_4_hr_left = patches_4_hr_left.cpu().detach().numpy()
        patches_4_hr_right = patches_4_hr_right.cpu().detach().numpy()
        hr_final_data = np.stack([patches_4_hr_left, patches_4_hr_right])
        lr_final_data = np.stack([patches_4_lr_left, patches_4_lr_right])
        patch_counter = 0
        npz_counter += 1
        os.mkdir(data_npz_dir+str(npz_counter))
        np.savez_compressed(data_npz_dir+str(npz_counter)+'/hr', hr_final_data)
        np.savez_compressed(data_npz_dir+str(npz_counter)+'/lr', lr_final_data)

        patches_4_lr_left = torch.empty(4, 3,5,32,88)
        patches_4_lr_right = torch.empty(4, 3,5,32,88)
        patches_4_hr_left = torch.empty(4, 3,32*4,88*4)
        patches_4_hr_right = torch.empty(4, 3,32*4,88*4)


    # if(patch_counter==4):
    #     patches_4_lr_left = patches_4_lr_left
    #     patches_4_lr_right = patches_4_lr_right
    #     patches_4_hr_left = patches_4_hr_left
    #     patches_4_hr_right = patches_4_hr_right
    #     hr_final_data = torch.stack([patches_4_hr_left, patches_4_hr_right])
    #     lr_final_data = torch.stack([patches_4_lr_left, patches_4_lr_right])
    #     patch_counter = 0
    #     npz_counter += 1
    #     os.mkdir(data_npz_dir+str(npz_counter))
    #     torch.save(hr_final_data, data_npz_dir+str(npz_counter)+'/hr.pt')
    #     torch.save(lr_final_data, data_npz_dir+str(npz_counter)+'/lr.pt')

    #     patches_4_lr_left = torch.empty(4, 3,5,32,88)
    #     patches_4_lr_right = torch.empty(4, 3,5,32,88)
    #     patches_4_hr_left = torch.empty(4, 3,32*4,88*4)
    #     patches_4_hr_right = torch.empty(4, 3,32*4,88*4)