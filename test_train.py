from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import os
# from model import *
from model_simple_ff import *
from torchvision import transforms
import cv2

from math import log10, sqrt



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
    parser.add_argument('--testset_dir', type=str, default='./data/train/')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_name', type=str, default='TransSNSR_4xSR_epoch1ablationFF')
    return parser.parse_args()


def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)

def test(cfg):
    # cap = cv2.VideoCapture('D:/copy_drive_c/data_64/training_data/Blur_1_l_3.mov')
    # ret, frame = cap.read()
    
    # # frame = cv2.resize(frame, (88*2,32*2)) 
    # frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25) 
    # cv2.imwrite("lr0.png", frame)


    spatial_dim = (int(1920/4), int(1080/4))
    net = Net(cfg.scale_factor, spatial_dim, cfg).to(cfg.device)
    model = torch.load('./log/' + cfg.model_name + '.pth.tar')
    net.load_state_dict(model['state_dict'])
    net.eval()

    file_list = os.listdir(cfg.testset_dir + '/patches_x' + str(cfg.scale_factor))
    for idx in range(len(file_list)):
        # img_lr_left_list = torch.empty(1,3,5,270,480).to(cfg.device)
        # img_lr_right_list = torch.empty(1,3,5,270,480).to(cfg.device)
        img_lr_left_list = torch.empty(1,3,5,32,88).to(cfg.device)
        img_lr_right_list = torch.empty(1,3,5,32,88).to(cfg.device)
        # img_lr_left_list = torch.empty(1,3,5,540,960).to(cfg.device)
        # img_lr_right_list = torch.empty(1,3,5,540,960).to(cfg.device)
        for i in range(1,6):
            LR_left = Image.open(cfg.testset_dir + '/patches_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/'+str(i)+ '/lr0.png')
            LR_right = Image.open(cfg.testset_dir  + '/patches_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/'+str(i)+ '/lr1.png')
            LR_left  = np.array(LR_left,  dtype=np.float32)
            LR_right = np.array(LR_right, dtype=np.float32)
                
            ll = toTensor(LR_left)
            rr = toTensor(LR_right)

            # LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
            # LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)
            img_lr_left_list[:,:,i-1,:,:], img_lr_right_list[:,:,i-1,:,:] = Variable(ll).to(cfg.device), Variable(rr).to(cfg.device)
        scene_name = file_list[idx]
        print('Running Scene ' + scene_name + ' of ' + ' Dataset......')

        # HR_left, HR_right, LR_left, LR_right  = Variable(HR_left).to(cfg.device), Variable(HR_right).to(cfg.device),\
        #                                         Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)

        with torch.no_grad():
            SR_left, SR_right = net(img_lr_left_list, img_lr_right_list, is_training=1)
            SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)
        save_path = './results/' + cfg.model_name + '/' + cfg.dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))

        hr0_path = cfg.testset_dir  + '/patches_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/'+str(3)+ '/hr0.png'
        hr1_path = cfg.testset_dir  + '/patches_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/'+str(3)+ '/hr1.png'
        hr0 = Image.open(hr0_path)
        hr0 = np.array(hr0,  dtype=np.float32)
        hr0 = Variable(toTensor(hr0))

        hr1 = Image.open(hr1_path)
        hr1 = np.array(hr1,  dtype=np.float32)
        hr1 = Variable(toTensor(hr1))

        psnr_value0 = PSNR(hr0.cpu().detach().numpy(), SR_left.cpu().detach().numpy())
        psnr_value1 = PSNR(hr1.cpu().detach().numpy(), SR_right.cpu().detach().numpy())
        print("psnr_value: ", (psnr_value0+psnr_value1)/2)



        SR_left_img.save(save_path + '/' + scene_name + '_L.png')
        SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
        SR_right_img.save(save_path + '/' + scene_name + '_R.png')


if __name__ == '__main__':
    cfg = parse_args()
    dataset_list = ['Flickr1024']
    for i in range(len(dataset_list)):
        cfg.dataset = dataset_list[i]
        test(cfg)
    print('Finished!')
