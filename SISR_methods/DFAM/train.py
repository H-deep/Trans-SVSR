import os
import time
import torch
import random
import argparse
import importlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *
from test import test, test_SISR
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='PyTorch Train')
parser.add_argument('--model', type=str, default='VDSR_DFAM', help='The name of model.')
parser.add_argument('--scale', type=int, default=4, help='upscale factor (default: 4)')
parser.add_argument('--batchSize', type=int, default=3, help='Training batch size.')
parser.add_argument('--nEpochs', type=int, default=30, help='Number of epochs to train')
parser.add_argument('--n_steps', type=int, default=100, help='initial LR decayed by momentum every n epochs')
parser.add_argument('--loss', type=str, default='all', help='all/no_smooth/no_smooth_photometric')
parser.add_argument('--upsample', action='store_true', default=False)
parser.add_argument('--rgb2y', action='store_true', default=False)
parser.add_argument('--pretrained', type=str, default='', help='path to pretrained')
parser.add_argument('--resume', type=str, default='', help='Path to pretrained SISR model (default: none)')
parser.add_argument('--start-epoch', type=int, default=1, help='Manual epoch number (useful on restarts)')
parser.add_argument('--trainset_dir', type=str, default='../data/train/Flickr1024_patches', help='')
parser.add_argument('--threads', type=int, default=2, help='Number of threads for data loader to use, Default: 1')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--gamma', type=float, default=0.5, help='')
parser.add_argument('--clip', type=float, default=0.4, help='Clipping Gradients. Default=0.4')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum, Default: 0.9')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay, Default: 1e-4')
parser.add_argument('--device', type=str, default='cuda')


def main():
    opt = parser.parse_args()

    out_path = os.path.join('ckpt', opt.model)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    logger = get_logger(os.path.join(out_path, opt.model + '_x' + str(opt.scale) + '_log.txt'))
    logger.info(opt)

    opt.seed = random.randint(1, 10000)
    logger.info('===> Random Seed: {}'.format(opt.seed))
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    # SummaryWriter
    writer = SummaryWriter(logdir=out_path)

    logger.info('===> Loading datasets')
    opt.dataset_dir = opt.trainset_dir
    train_set = TrainSetLoader(opt, logger)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    logger.info('===> Building model')
    net = get_model(opt)
    net.apply(weights_init_xavier)

    logger.info('===> Setting Optimizer')
    optimizer = optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.n_steps, gamma=opt.gamma)

    # optionally train from a pretrained SISR
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            logger.info('===> loading checkpoint {}'.format(opt.pretrained))
            net_dict = net.state_dict()
            if 'RCAN' in opt.model:
                pretrained_dict = torch.load(opt.pretrained)
            else:
                pretrained_dict = torch.load(opt.pretrained)['model']
            update_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            net_dict.update(update_dict)
            net.load_state_dict(net_dict)
        else:
            logger.info('===> no checkpoint found at {}'.format(opt.resume))
    else:
        logger.info('===> opt.pretrained is null')

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info('===> loading model {}'.format(opt.resume))
            state = torch.load(opt.resume)
            opt.start_epoch = state["epoch"] + 1
            net.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
        else:
            logger.info('===> no model found at {}'.format(opt.resume))
    else:
        logger.info('===> opt.resume is null')
    net.eval()

    # # add graph
    # if opt.rgb2y:
    #     writer.add_graph(net, (torch.ones(opt.batchSize, 1, 120, 360).cuda(), torch.ones(opt.batchSize, 1, 120, 360).cuda()))
    # else:
    #     writer.add_graph(net, (torch.ones(opt.batchSize, 3, 120, 360).cuda(), torch.ones(opt.batchSize, 3, 120, 360).cuda()))

    logger.info('===> Training')
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        if 'DFAM' in opt.model:
            train_DFAM(opt, train_loader, net, scheduler, optimizer, epoch, writer, out_path, logger)
            for test_data in ['middlebury', 'KITTI2012', 'KITTI2015']:
                logger.info('===> test data {} start!'.format(test_data))
                opt.dataset_dir = os.path.join('../data/test', test_data)
                test_set = TestSetLoader(opt, logger)
                test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
                test(opt, test_loader, net, logger, None)
                logger.info('\n')
        elif 'SAM' in opt.model:
            train_SAM(opt, train_loader, net, scheduler, optimizer, epoch, writer, out_path, logger)
        else:
            train_SISR(opt, train_loader, net, scheduler, optimizer, epoch, writer, out_path, logger)
            for test_data in ['middlebury', 'KITTI2012', 'KITTI2015']:
                logger.info('===> test data {} start!'.format(test_data))
                opt.dataset_dir = os.path.join('../data/test', test_data)
                test_set = TestSetLoader(opt, logger)
                test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
                test_SISR(opt, test_loader, net, logger, None)
                logger.info('\n')


def train_DFAM(opt, train_loader, net, scheduler, optimizer, epoch, writer, out_path, logger):
    logger.info('Epoch = {}, lr = {}'.format(epoch, optimizer.param_groups[0]["lr"]))
    net.train()

    criterion_L1 = nn.L1Loss().to(opt.device)
    criterion_mse = nn.MSELoss(reduction='sum').to(opt.device)

    loss_epoch = 0.0
    psnr_epoch = 0.0
    start_time = time.time()
    for iteration, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
        if opt.rgb2y:  # VDSR, SRCNN
            HR_left, HR_right, LR_left, LR_right = \
                rgb2y(HR_left) / 255, rgb2y(HR_right) / 255, rgb2y(LR_left) / 255, rgb2y(LR_right) / 255
        else:  # SRResNet
            HR_left, HR_right, LR_left, LR_right = HR_left / 255, HR_right / 255, LR_left / 255, LR_right / 255

        b, c, h, w = LR_left.size()
        HR_left, HR_right, LR_left, LR_right = \
            Variable(HR_left).to(opt.device), Variable(HR_right).to(opt.device), \
            Variable(LR_left).to(opt.device), Variable(LR_right).to(opt.device)

        SR_left, SR_right, p_list, shift_list = net(LR_left, LR_right)

        # loss_SR
        loss = criterion_mse(SR_left, HR_left)

        for i in range(0, len(p_list), 2):
            # loss_photometric
            if p_list[i] is not None and opt.loss != 'no_smooth_photometric':
                LR_p_left = get_x_p(LR_left, p_list[i])
                LR_p_right = get_x_p(LR_right, p_list[i+1])
                loss_photo_left = criterion_L1(LR_left, LR_p_right)
                loss_photo_right = criterion_L1(LR_right, LR_p_left)
                loss = loss + 0.01 * (loss_photo_left + loss_photo_right)
            # loss_disparity
            if shift_list[i] is not None and opt.loss != 'no_smooth' and opt.loss != 'no_smooth_photometric':
                shift_left = torch.mean(shift_list[i]).expand_as(shift_list[i])
                shift_right = torch.mean(shift_list[i+1]).expand_as(shift_list[i+1])
                loss_shift_left = criterion_L1(shift_list[i], shift_left)
                loss_shift_right = criterion_L1(shift_list[i+1], shift_right)
                loss = loss + 0.0001 * (loss_shift_left + loss_shift_right)

        loss_epoch = loss_epoch + loss.detach().item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), opt.clip)
        optimizer.step()

        psnr = cal_psnr(HR_left.cpu().detach().numpy(), SR_left.cpu().detach().numpy())
        psnr_epoch = psnr_epoch + psnr

        writer.add_scalar(opt.model + '_x' + str(opt.scale) + '_loss', loss, epoch * len(train_loader) + iteration)
        writer.add_scalar(opt.model + '_x' + str(opt.scale) + '_psnr', psnr, epoch * len(train_loader) + iteration)

        if (iteration + 1) % 1000 == 0:
            end_time = time.time()
            logger.info('iteration/total: {}/{}  loss:{:.3f} psnr: {:.3f}  time: {:.3f}'.format(
                iteration + 1, len(train_loader), loss_epoch / (iteration + 1), psnr_epoch / (iteration + 1), end_time - start_time))

    scheduler.step()
    save_checkpoint(epoch, net, optimizer, scheduler, out_path,
                    opt.model + '_x' + str(opt.scale) + '_epoch' + str(epoch) + '.pth', logger)

    logger.info('===> Epoch[{}]: Loss: {:.3f} PSNR: {:.3f}\n'.format(epoch, loss_epoch / (iteration + 1), psnr_epoch / (iteration + 1)))


def train_SAM(opt, train_loader, net, scheduler, optimizer, epoch, writer, out_path, logger):

    logger.info('Epoch = {}, lr = {}'.format(epoch, optimizer.param_groups[0]["lr"]))
    net.train()

    loss_epoch = 0.0
    psnr_epoch = 0.0
    criterion_L1 = nn.L1Loss().to(opt.device)
    criterion_mse = nn.MSELoss(reduction='sum').to(opt.device)

    start_time = time.time()
    for iteration, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
        LR_left, LR_right, HR_left, HR_right = rgb2y(LR_left) / 255, rgb2y(LR_right) / 255, rgb2y(HR_left) / 255, rgb2y(HR_right) / 255
        b, c, h, w = LR_left.shape
        LR_left, LR_right, HR_left, HR_right = Variable(LR_left).to(opt.device), Variable(LR_right).to(opt.device), \
                                               Variable(HR_left).to(opt.device), Variable(HR_right).to(opt.device)

        HR_l, HR_r, map, mask = net(LR_left, LR_right)
        (M_right_to_left0, M_left_to_right0) = map[0]
        (M_right_to_left1, M_left_to_right1) = map[1]
        (V_right_to_left0, V_left_to_right0) = mask[0]
        (V_right_to_left1, V_left_to_right1) = mask[1]
        ###loss_SR

        loss_SR = criterion_mse(HR_l, HR_left)

        ### loss_photometric0
        LR_right_warped = torch.bmm(M_right_to_left0.contiguous().view(b * h, w, w),
                                    LR_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        LR_left_warped = torch.bmm(M_left_to_right0.contiguous().view(b * h, w, w),
                                   LR_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

        loss_photo0 = criterion_L1(LR_left * V_left_to_right0, LR_right_warped * V_left_to_right0) + \
                     criterion_L1(LR_right * V_right_to_left0, LR_left_warped * V_right_to_left0)

        ### loss_photometric1
        LR_right_warped = torch.bmm(M_right_to_left1.contiguous().view(b * h, w, w),
                                    LR_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        LR_left_warped = torch.bmm(M_left_to_right1.contiguous().view(b * h, w, w),
                                   LR_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

        loss_photo1 = criterion_L1(LR_left * V_left_to_right1, LR_right_warped * V_left_to_right1) + \
                      criterion_L1(LR_right * V_right_to_left1, LR_left_warped * V_right_to_left1)

        # print(loss_SR, loss_photo0 + loss_photo1)
        loss = loss_SR + 0.01 * (loss_photo0 + loss_photo1)

        loss_epoch = loss_epoch + loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), opt.clip)
        optimizer.step()
        PSNR = cal_psnr(HR_left.cpu().detach().numpy(), HR_l.cpu().detach().numpy())
        psnr_epoch = psnr_epoch + PSNR

        if (iteration+1) % 1000 == 0:
            end_time = time.time()
            logger.info('iteration/total: {}/{}  PSNR: {}  time: {}'.format(iteration+1, len(train_loader), psnr_epoch/(iteration+1), end_time-start_time))

    scheduler.step()
    save_checkpoint(epoch, net, optimizer, scheduler, out_path,
                    opt.model + '_x' + str(opt.scale) + '_epoch' + str(epoch) + '.pth', logger)

    logger.info('===> Epoch[{}]: Loss: {:.3f} PSNR: {:.3f}\n'.format(epoch, loss_epoch/(iteration+1), psnr_epoch/(iteration+1)))


def train_SISR(opt, train_loader, net, scheduler, optimizer, epoch, writer, out_path, logger):
    logger.info('Epoch = {}, lr = {}'.format(epoch, optimizer.param_groups[0]["lr"]))
    net.train()

    criterion_L1 = nn.L1Loss().to(opt.device)
    criterion_mse = nn.MSELoss(reduction='sum').to(opt.device)

    loss_epoch = 0.0
    psnr_epoch = 0.0
    start_time = time.time()
    for iteration, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
        if opt.rgb2y:  # VDSR, SRDenseNet, SRCNN
            HR_left, HR_right, LR_left, LR_right = \
                rgb2y(HR_left) / 255, rgb2y(HR_right) / 255, rgb2y(LR_left) / 255, rgb2y(LR_right) / 255
        else:  # SRResNet, RCAN
            HR_left, HR_right, LR_left, LR_right = HR_left / 255, HR_right / 255, LR_left / 255, LR_right / 255

        b, c, h, w = LR_left.size()
        HR_left, HR_right, LR_left, LR_right = \
            Variable(HR_left).to(opt.device), Variable(HR_right).to(opt.device), \
            Variable(LR_left).to(opt.device), Variable(LR_right).to(opt.device)

        if random.random() < 0.5:
            LR_left = LR_right
            HR_left = HR_right

        SR_left = net(LR_left)

        ###loss_SR
        loss_SR = criterion_mse(SR_left, HR_left)
        loss = loss_SR
        loss_epoch = loss_epoch + loss.detach().item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), opt.clip)
        optimizer.step()

        psnr = cal_psnr(HR_left.cpu().detach().numpy(), SR_left.cpu().detach().numpy())
        psnr_epoch = psnr_epoch + psnr

        writer.add_scalar(opt.model + '_x' + str(opt.scale) + '_loss', loss, epoch * len(train_loader) + iteration)
        writer.add_scalar(opt.model + '_x' + str(opt.scale) + '_psnr', psnr, epoch * len(train_loader) + iteration)

        if (iteration + 1) % 1000 == 0:
            end_time = time.time()
            logger.info('iteration/total: {}/{}  loss:{:.3f} psnr: {:.3f}  time: {:.3f}'.format(
                iteration + 1, len(train_loader), loss_epoch / (iteration + 1), psnr_epoch / (iteration + 1),
                end_time - start_time))

    scheduler.step()
    save_checkpoint(epoch, net, optimizer, scheduler, out_path,
                    opt.model + '_x' + str(opt.scale) + '_epoch' + str(epoch) + '.pth', logger)

    logger.info('===> Epoch[{}]: Loss: {:.3f} PSNR: {:.3f}\n'.format(epoch, loss_epoch / (iteration + 1), psnr_epoch / (iteration + 1)))


if __name__ == '__main__':
    main()

