import os

import numpy as np
import requests
import torch


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr
        lr = param_group['lr']
    return lr


def opt_save(opt):
    log_path = opt.save_path + "train_settings.log"
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    att = [i for i in opt.__dir__() if not i.startswith("_")]
    with open(log_path, "w") as f:
        for i in att:
            print("{}:{}".format(i, eval(f"opt.{i}")), file=f)


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def fps(model, epoch_num, size):
    ls = []  # 每次计算得到的fps
    iterations = 300  # 重复计算的轮次
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    model.to(device)

    random_input = torch.randn(1, 3, size, size).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU预热
    for _ in range(50):
        _ = model(random_input, random_input)

    for i in range(epoch_num):
        # 测速
        times = torch.zeros(iterations)  # 存储每轮iteration的时间
        with torch.no_grad():
            for iter in range(iterations):
                starter.record()
                _ = model(random_input, random_input)
                ender.record()
                # 同步GPU时间
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)  # 计算时间
                times[iter] = curr_time
                # print(curr_time)

        mean_time = times.mean().item()
        ls.append(1000 / mean_time)
        print("{}/{} Inference time: {:.6f}, FPS: {} ".format(i + 1, epoch_num, mean_time, 1000 / mean_time))
    print(f"平均fps为 {np.mean(ls):.2f}")