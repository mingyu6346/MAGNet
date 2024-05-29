import logging
import sys
import torch.nn.functional as F
import torch
import numpy as np
import os, argparse
from datetime import datetime
from tqdm import tqdm
from model.MAGNet import MAGNet
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr, opt_save, iou_loss

import pytorch_iou

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training image size')
parser.add_argument('--continue_train', type=bool, default=False, help='continue training')
parser.add_argument('--continue_train_path', type=str, default='', help='continue training path')

parser.add_argument('--rgb_root', type=str, default='D:/DataSet/SOD/RGB-D SOD/train_dut/RGB/',
                    help='the training rgb images root')  # train_dut
parser.add_argument('--depth_root', type=str, default='D:/DataSet/SOD/RGB-D SOD/train_dut/depth/',
                    help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='D:/DataSet/SOD/RGB-D SOD/train_dut/GT/',
                    help='the training gt images root')

parser.add_argument('--val_rgb', type=str, default="D:/DataSet/SOD/RGB-D SOD/test_data/NLPR/RGB/",
                    help='validate rgb path')
parser.add_argument('--val_depth', type=str, default="D:/DataSet/SOD/RGB-D SOD/test_data/NLPR/depth/",
                    help='validate depth path')
parser.add_argument('--val_gt', type=str, default="D:/DataSet/SOD/RGB-D SOD/test_data/NLPR/GT/",
                    help='validate gt path')

parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
parser.add_argument('--save_path', type=str, default="ckps/MAGNet/", help='checkpoint path')

opt = parser.parse_args()

opt_save(opt)
logging.basicConfig(filename=opt.save_path + 'log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %H:%M:%S %p')
logging.info("Net-Train")
# model
model = MAGNet()
if os.path.exists("ckps/smt/smt_tiny.pth"):
    model.rgb_backbone.load_state_dict(torch.load("ckps/smt/smt_tiny.pth")['model'])
    print(f"loaded imagenet pretrained SMT from ckps/smt/smt_tiny.pth")
else:
    raise "please put smt_tiny.pth under ckps/smt/ folder"
if opt.continue_train:
    model.load_state_dict(torch.load(opt.continue_train_path))
    print(f"continue training from {opt.continue_train_path}")

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

print('load data...')
train_loader = get_loader(opt.rgb_root, opt.gt_root, opt.depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average=True)
best_loss = 1.0


def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_list = []
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        (images, gts, depth) = pack
        images = images.cuda()
        gts = gts.cuda()
        depth = depth.cuda().repeat(1, 3, 1, 1)

        pred_1, pred_2, pred_3, pred_4 = model(images, depth)

        loss1 = CE(pred_1, gts) + iou_loss(pred_1, gts)
        loss2 = CE(pred_2, gts) + iou_loss(pred_2, gts)
        loss3 = CE(pred_3, gts) + iou_loss(pred_3, gts)
        loss4 = CE(pred_4, gts) + iou_loss(pred_4, gts)

        loss = loss1 + loss2 + loss3 + loss4

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        loss_list.append(float(loss))
        if i % 20 == 0 or i == total_step:
            msg = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.format(
                datetime.now(), epoch, opt.epoch, i, total_step,
                opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data,
                loss2.data, loss3.data, loss4.data)
            print(msg)
            logging.info(msg)
    epoch_loss = np.mean(loss_list)
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    global best_loss
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), opt.save_path + 'Net_rgb_d.pth' + f'.{epoch}_{epoch_loss:.3f}',
                   _use_new_zipfile_serialization=False)
    with open(opt.save_path + "loss.log", "a") as f:
        print(f"{datetime.now()}  epoch {epoch}  loss {np.mean(loss_list):.3f}", file=f)


best_mae = 1.0
best_epoch = 0


def validate(test_dataset, model, epoch, opt):
    global best_mae, best_epoch
    model.eval().cuda()
    mae_sum = 0
    test_loader = test_dataset(opt.val_rgb, opt.val_gt, opt.val_depth, opt.trainsize)
    with torch.no_grad():
        for i in tqdm(range(test_loader.size), desc="Validating", file=sys.stdout):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()

            res, _, _, _ = model(image, depth)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()

            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # mae_train = torch.sum(torch.abs(res - gt)) * 1.0 / (torch.numel(gt))
            mae_train = np.mean(np.abs(res - gt))
            mae_sum = mae_train + mae_sum
    mae = mae_sum / test_loader.size

    if epoch == 0:
        best_mae = mae
    else:
        if mae < best_mae:
            best_mae = round(mae, 5)
            best_epoch = epoch
            torch.save(model.state_dict(), opt.save_path + 'MAGNet_mae_best.pth', _use_new_zipfile_serialization=False)
            print('best epoch:{}'.format(epoch))
    msg = 'Epoch: {} MAE: {:.5f} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch)
    print(msg)
    logging.info(msg)
    with open(f"{opt.save_path}mae.log", "a", encoding='utf-8') as f:
        f.write('Epoch: {:03d} MAE: {:.5f} ####  bestMAE: {:.5f} bestEpoch: {:03d}\n'.format(epoch, mae, best_mae,
                                                                                             best_epoch))
    return mae


print("Let's go!")
for epoch in range(opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
    validate(test_dataset, model, epoch, opt)
