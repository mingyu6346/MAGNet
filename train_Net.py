import logging
import torch
import numpy as np
import os, argparse
from datetime import datetime
from model.MAGNet import MAGNet
from data import get_loader
from utils import clip_gradient, adjust_lr, iou_loss

import pytorch_iou


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training image size')
parser.add_argument('--continue_train', type=bool, default=False, help='continue training')
parser.add_argument('--continue_train_path', type=str, default='', help='continue training path')
parser.add_argument('--rgb_root', type=str, default='',
                    help='the training rgb images root')  # train_dut
parser.add_argument('--depth_root', type=str, default='',
                    help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='',
                    help='the training gt images root')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
parser.add_argument('--save_path', type=str, default="", help='checkpoint path')

opt = parser.parse_args()


logging.basicConfig(filename=opt.save_path + 'log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %H:%M:%S %p')
logging.info("Net-Train")
# model
model = MAGNet()
model.rgb_backbone.load_state_dict(torch.load("ckps/smt/smt_tiny.pth")['model'])
if opt.continue_train:
    model.load_state_dict(torch.load(opt.continue_train_path))

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


print("Let's go!")
for epoch in range(opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
