from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import os
# from model import *
from model_simple import *
from torchvision import transforms
import cv2

from math import log10, sqrt
from skimage.metrics import structural_similarity as compare_ssim
import random


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    # max_pixel = 255.0
    try:
        psnr = 20 * log10(max_pixel / sqrt(mse))
    except:
        psnr = 38
    return psnr

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--testset_dir', type=str, default='./data/test/SVSRset/lr_x4/')
    parser.add_argument('--testset_dir', type=str, default='./data/test/')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='TransSVSR_4xSR')
    return parser.parse_args()


def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)

def test(cfg):
    spatial_dim = (int(1920/4), int(1080/4))
    net = Net(cfg.scale_factor, spatial_dim, cfg).to(cfg.device)
    model = torch.load('./log2/' + cfg.model_name + '.pth.tar')
    for m in net.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm2d:
                child.track_runing_stats = False
                child.runing_mean = None
                child.runing_var = None

    for m in net.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm3d:
                child.track_runing_stats = False
                child.runing_mean = None
                child.runing_var = None
    # net.load_state_dict(model['state_dict'])
    net.eval()


    file_list = os.listdir(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor))
    random.shuffle(file_list)


    psnr = 0
    score = 0

    fr_counter = 0

    for idx in range(len(file_list)):
        img_lr_left_list = torch.empty(1,3,5,270,480).to(cfg.device)
        img_lr_right_list = torch.empty(1,3,5,270,480).to(cfg.device)
        # img_lr_left_list = torch.empty(1,3,5,180,320).to(cfg.device)
        # img_lr_right_list = torch.empty(1,3,5,180,320).to(cfg.device)
        
        for i in range(1,6):
            LR_left = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/'+str(i)+ '/lr0.png')
            LR_right = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/'+str(i)+ '/lr1.png')
            LR_left  = np.array(LR_left,  dtype=np.float32)
            LR_right = np.array(LR_right, dtype=np.float32)
                
            ll = toTensor(LR_left)
            rr = toTensor(LR_right)

            img_lr_left_list[:,:,i-1,:,:], img_lr_right_list[:,:,i-1,:,:] = Variable(ll).to(cfg.device), Variable(rr).to(cfg.device)
        scene_name = file_list[idx]
        print('Running Scene ' + scene_name + ' of ' + cfg.dataset + ' Dataset......')

        if fr_counter<10:
            net.train()
        else:
            net.train()

        with torch.no_grad():
            SR_left, SR_right = net(img_lr_left_list, img_lr_right_list, is_training=1)
            SR_left, SR_right = torch.clamp(SR_left[0], 0, 1), torch.clamp(SR_right[0], 0, 1)
        save_path = './results/' + cfg.model_name + '/' + cfg.dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))

        hr0_path = cfg.testset_dir + cfg.dataset + '/hr' + '/'+file_list[idx]+ '/hr0.png'
        hr1_path = cfg.testset_dir + cfg.dataset + '/hr' + '/'+file_list[idx]+ '/hr1.png'
        hr0 = Image.open(hr0_path)
        hr0 = np.array(hr0,  dtype=np.float32)
        hr0 = Variable(toTensor(hr0))

        hr1 = Image.open(hr1_path)
        hr1 = np.array(hr1,  dtype=np.float32)
        hr1 = Variable(toTensor(hr1))

        psnr_value0 = PSNR(hr0.cpu().detach().numpy(), SR_left.cpu().detach().numpy())
        psnr_value1 = PSNR(hr1.cpu().detach().numpy(), SR_right.cpu().detach().numpy())
        # print("psnr_value: ", (psnr_value0+psnr_value1)/2)
        psnr += (psnr_value0+psnr_value1)/2


        grayA = cv2.cvtColor(hr0.permute(1,2,0).cpu().detach().numpy(), cv2.COLOR_BGR2GRAY)
        grayAA = cv2.cvtColor(hr1.permute(1,2,0).cpu().detach().numpy(), cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(SR_left.permute(1,2,0).cpu().detach().numpy(), cv2.COLOR_BGR2GRAY)
        grayBB = cv2.cvtColor(SR_right.permute(1,2,0).cpu().detach().numpy(), cv2.COLOR_BGR2GRAY)
        # multichannel=True
        score0 = compare_ssim(grayA, grayB, data_range=grayA.max() - grayA.min())
        score2 = compare_ssim(grayAA, grayBB, data_range=grayAA.max() - grayAA.min())
        score += (score0+score2)/2


        fr_counter += 1
        final_psnr = psnr / fr_counter
        print("psnr_value: ", (psnr_value0))
        print("avg_psnr_value: ", final_psnr)


        final_ssim = score / fr_counter
        print("ssim_value: ", score0)
        print("avg_ssim_value: ", final_ssim)
        print("fr_counter:", fr_counter)

        if fr_counter == 1:
            os.mkdir(save_path + '/left')
            os.mkdir(save_path + '/right')
            os.mkdir(save_path + '/left_hr')
            os.mkdir(save_path + '/right_hr')


        SR_left_img.save(save_path + '/left' + '/' + scene_name + '_L.png')
        SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
        SR_right_img.save(save_path + '/right' + '/' + scene_name + '_R.png')

        # cv2.imwrite(save_path + '/left_hr' + '/' + scene_name + '_L.png', hr0.permute(1,2,0).cpu().detach().numpy())
        hr00 = Image.open(hr0_path)
        hr11 = Image.open(hr1_path)

        hr00.save(save_path + '/left_hr' + '/' + scene_name + '_L.png')
        hr11.save(save_path + '/right_hr' + '/' + scene_name + '_R.png')
        # hr1.cpu().detach().numpy().save(save_path + '/right_hr' + '/' + scene_name + '_L.png')


if __name__ == '__main__':
    cfg = parse_args()
    dataset_list = ['SVSRset']
    for i in range(len(dataset_list)):
        cfg.dataset = dataset_list[i]
        test(cfg)
    print('Finished!')
