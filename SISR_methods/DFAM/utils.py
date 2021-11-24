import os
import torch
import psutil
import pynvml
import random
import logging
import importlib
import numpy as np
import torch.nn as nn
from pynvml import *
from PIL import Image
from math import sqrt
# from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor, ToPILImage


class TrainSetLoader(Dataset):
    def __init__(self, opt, logger):
        super(TrainSetLoader, self).__init__()
        self.opt = opt
        self.dataset_dir = opt.dataset_dir
        self.scale = opt.scale
        self.model = opt.model
        self.file_list = os.listdir(self.dataset_dir + '/patches_x' + str(self.scale))
        self.file_list.sort()
        logger.info('===> The number of train dataset = {}'.format(len(self.file_list)))

    def __getitem__(self, index):
        hr_image_left = Image.open(self.dataset_dir + '/patches_x' + str(self.scale) + '/' + self.file_list[index] + '/hr0.png')
        hr_image_right = Image.open(self.dataset_dir + '/patches_x' + str(self.scale) + '/' + self.file_list[index] + '/hr1.png')
        lr_image_left = Image.open(self.dataset_dir + '/patches_x' + str(self.scale) + '/' + self.file_list[index] + '/lr0.png')
        lr_image_right = Image.open(self.dataset_dir + '/patches_x' + str(self.scale) + '/' + self.file_list[index] + '/lr1.png')

        hr_image_left = np.array(hr_image_left, dtype=np.float32)
        hr_image_right = np.array(hr_image_right, dtype=np.float32)
        if self.opt.upsample:  # VDSR, SRCNN
            w, h, c = hr_image_left.shape
            lr_image_left = np.array(lr_image_left.resize((h, w), Image.BICUBIC), dtype=np.float32)
            lr_image_right = np.array(lr_image_right.resize((h, w), Image.BICUBIC), dtype=np.float32)
            hr_image_left, hr_image_right, lr_image_left, lr_image_right = \
                augumentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right)
        else:  # SRResNet, RCAN
            lr_image_left = np.array(lr_image_left, dtype=np.float32)
            lr_image_right = np.array(lr_image_right, dtype=np.float32)
            hr_image_left, hr_image_right, lr_image_left, lr_image_right = \
                augumentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right)

        return ToTensor()(hr_image_left), ToTensor()(hr_image_right), ToTensor()(lr_image_left), ToTensor()(lr_image_right)

    def __len__(self):
        return len(self.file_list)


class TestSetLoader(Dataset):
    def __init__(self, opt, logger):
        super(TestSetLoader, self).__init__()
        self.opt = opt
        self.dataset_dir = opt.dataset_dir
        self.scale = opt.scale
        self.model = opt.model
        self.file_list = os.listdir(self.dataset_dir + '/hr')
        self.file_list.sort()
        logger.info('===> The number of test dataset = {}'.format(len(self.file_list)))

    def __getitem__(self, index):
        hr_image_left = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr0.png')
        hr_image_right = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr1.png')
        lr_image_left = Image.open(self.dataset_dir + '/lr_x' + str(self.scale) + '/' + self.file_list[index] +'/3'+ '/lr0.png')
        lr_image_right = Image.open(self.dataset_dir + '/lr_x' + str(self.scale) + '/' + self.file_list[index] +'/3'+ '/lr1.png')

        hr_image_left = np.array(hr_image_left, dtype=np.float32)
        hr_image_right = np.array(hr_image_right, dtype=np.float32)
        if self.opt.upsample:  # VDSR, SRCNN
            w, h, c = hr_image_left.shape
            lr_image_left = np.array(lr_image_left.resize((h, w), Image.BICUBIC), dtype=np.float32)
            lr_image_right = np.array(lr_image_right.resize((h, w), Image.BICUBIC), dtype=np.float32)
        else:  # SRResNet, RCAN
            lr_image_left = np.array(lr_image_left, dtype=np.float32)
            lr_image_right = np.array(lr_image_right, dtype=np.float32)

        return ToTensor()(hr_image_left), ToTensor()(hr_image_right), ToTensor()(lr_image_left), ToTensor()(lr_image_right)

    def __len__(self):
        return len(self.file_list)


def augumentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
    if random.random() < 0.5:  # flip horizonly
        lr_image_left = lr_image_left[:, ::-1, :]
        lr_image_right = lr_image_right[:, ::-1, :]
        hr_image_left = hr_image_left[:, ::-1, :]
        hr_image_right = hr_image_right[:, ::-1, :]
        tmp = lr_image_left
        lr_image_left = lr_image_right
        lr_image_right = tmp
        tmp = hr_image_left
        hr_image_left = hr_image_right
        hr_image_right = tmp
    if random.random() < 0.5:  # flip vertically
        lr_image_left = lr_image_left[::-1, :, :]
        lr_image_right = lr_image_right[::-1, :, :]
        hr_image_left = hr_image_left[::-1, :, :]
        hr_image_right = hr_image_right[::-1, :, :]
    """"no rotation
    if random.random()<0.5:
        lr_image_left = lr_image_left.transpose(1, 0, 2)
        lr_image_right = lr_image_right.transpose(1, 0, 2)
        hr_image_left = hr_image_left.transpose(1, 0, 2)
        hr_image_right = hr_image_right.transpose(1, 0, 2)
    """
    return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right),\
        np.ascontiguousarray(lr_image_left), np.ascontiguousarray(lr_image_right)


def cal_psnr(img1, img2):
    """
    Calculate psnr of the img1 and the img2.
    :param img1: numpy array
    :param img2: numpy array
    :return: np.float32
    """
    return measure.compare_psnr(img1, img2)


def cal_ssim(img1, img2):
    """
    Calculate ssim of the img1 and the img2.
    :param img1: numpy array
    :param img2: numpy array
    :return: np.float32
    """
    img1 = img1.transpose((1, 2, 0))
    img2 = img2.transpose((1, 2, 0))
    return compare_ssim(img1, img2, multichannel=True)


def save_checkpoint(epoch, net, optimizer, scheduler, save_path, file_name, logger):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    net = net.eval()
    state = {"epoch": epoch,
             "model": net.state_dict(),
             "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict()}
    path_file = os.path.join(save_path, file_name)
    torch.save(state, path_file)
    logger.info('Checkpoint saved to {}'.format(path_file))


# def toTensor(img):
#     img = torch.from_numpy(img.transpose((2, 0, 1)))
#     return img.float().div(255)


def save_img(img, path):  # TODO check
    '''
    :param img: a tensor, divided by 255
    :param path:
    :return:
    '''
    img = ToPILImage()(img)
    img.save(path)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def rgb2y(img):
    '''
    :param img: obtained by ToTensor, not divided by 255
    :return: a tensor, not divided by 255
    '''
    img_r = img[:, 0, :, :]
    img_g = img[:, 1, :, :]
    img_b = img[:, 2, :, :]
    image_y = torch.round(0.257 * torch.unsqueeze(img_r, 1) + 0.504 * torch.unsqueeze(img_g, 1) + 0.098 * torch.unsqueeze(img_b, 1) + 16)
    return image_y


def img_transfer(img, img_y):
    '''
    obtain a tensor of img_RBG by img_y
    :param img: a tensor obtained by Totensor, divided by 255
    :param img_y: a tensor, divided by 255
    :return:
    '''
    img_r = img[:, 0, :, :]
    img_g = img[:, 1, :, :]
    img_b = img[:, 2, :, :]
    image_y = torch.squeeze(img_y, 1)
    image_cb = (-0.148 * img_r - 0.291 * img_g + 0.439 * img_b + 128 / 255.0)
    image_cr = (0.439 * img_r - 0.368 * img_g - 0.071 * img_b + 128 / 255.0)
    image_r = (1.164 * torch.unsqueeze((image_y - 16/255), 1) + 1.596 * torch.unsqueeze((image_cr - 128/255), 1))
    image_g = (1.164 * torch.unsqueeze((image_y - 16/255), 1) - 0.392 * torch.unsqueeze((image_cb - 128/255), 1) - 0.813 * torch.unsqueeze((image_cr - 128/255), 1))
    image_b = (1.164 * torch.unsqueeze((image_y - 16/255), 1) + 2.017 * torch.unsqueeze((image_cb - 128/255), 1))
    image = torch.cat((image_r, image_g, image_b), 1)
    return image


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, sqrt(2. / n))
    elif isinstance(m, nn.Linear):
        n = m.in_features * m.out_features
        m.weight.data.normal_(0, sqrt(2. / n))


def get_model(opt):
    module = importlib.import_module('model.{}'.format(opt.model))
    if opt.model == 'VDSR':
        net = module.Net().to(opt.device)
    elif opt.model == 'VDSR_SAM':
        net = module.Net_SAM(n_intervals=[6, 12], n_blocks=18, inchannels=1, nfeats=64, outchannels=1).to(opt.device)
    elif opt.model == 'VDSR_DFAM':
        net = module.Net([5, 10, 15]).to(opt.device)
        print("VDSR_DFAM: ok")
    elif opt.model == 'SRCNN':
        net = module.Net().to(opt.device)
    elif opt.model == 'SRCNN_DFAM':
        net = module.Net().to(opt.device)
    elif opt.model == 'SRResNet':
        net = module.Net(n_blocks=16, scale=int(opt.scale)).to(opt.device)
    elif opt.model == 'SRResNet_DFAM':
        net = module.Net([6, 11], n_blocks=16, scale=int(opt.scale)).to(opt.device)
    elif opt.model == 'RCAN':
        net = module.Net(scale=int(opt.scale), n_resgroups=10, n_resblocks=20).to(opt.device)
    elif opt.model == 'RCAN_DFAM':
        net = module.Net([4, 7], scale=int(opt.scale), n_resgroups=10, n_resblocks=20).to(opt.device)
    return net


def get_memory_info():
    process = psutil.Process(os.getpid())
    memInfo = process.memory_info()
    print('内存占用: {:.2f}G'.format(memInfo.rss/1024/1024/1024))
    # 获取指定GPU显存信息
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU id
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('显存占用: {:.2f} G'.format(meminfo.used / 1024 / 1024 / 1024))  # 已用显存
    print('显存总共: {:.2f} G'.format(meminfo.total / 1024 / 1024 / 1024))  # 全部显存
    print('显存可用: {:.2f} G'.format(meminfo.free / 1024 / 1024 / 1024))  # 剩余显存


def get_gpu_info():
    # 获取GPU型号和驱动版本
    nvmlInit()
    print('Driver Version: {} utf-8'.format(str(nvmlSystemGetDriverVersion())))  # 显卡驱动版本
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print('Device {} : {} utf-8'.format(i, str(nvmlDeviceGetName(handle))))  # 每一个显卡的型号
    nvmlShutdown()


def get_x_p(x, p):
    b, h, w, _ = p.size()
    c = x.size(1)
    x = x.contiguous().view(b, c, -1)  # (b, c, h*w)
    index = p[..., 0] * w + p[..., 1]  # p_x*w+p_y, (b, h, w)
    index = index.unsqueeze(dim=1).expand(-1, c, -1, -1).contiguous().view(b, c, -1)  # (b, c, h*w)
    x_shift = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w)  # (b, c, h, w)
    return x_shift
