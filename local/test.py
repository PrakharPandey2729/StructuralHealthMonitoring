# coding=utf-8
import os
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model_class import CLASS_NET
from model_cover import COVER_NET
import util
import numpy as np
import cv2
import shutil
import torch.nn.functional as F
import math
import random

class_net = CLASS_NET()
class_net.cuda().eval()

cover_net = COVER_NET()
cover_net.cuda().eval()

ckpt_cover_path = "checkpoint/class_net_final.pth"
cover_net.load_state_dict(torch.load(ckpt_cover_path))
ckpt_class_path = "checkpoint/cover_net_final.pth"
class_net.load_state_dict(torch.load(ckpt_class_path))

batch = 1
criterion = nn.CrossEntropyLoss()
torch.set_grad_enabled(False)

data_path = './demo_data/Validation/'

transform = transforms.Compose([transforms.Resize((512, 512)),
transforms.ToTensor()])
train_set = datasets.ImageFolder(data_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=False, num_workers=2)

dataiter = iter(train_loader)
lenth = len(train_set)
print('img_num', lenth)
out_path = './view/'

if os.path.exists(out_path):
    shutil.rmtree(out_path)

save_num = 0
error_num = 0
break_flag = 0
area_add = 0
batch_num = 0
for images0, labels in dataiter:
    if break_flag:
        break
    batch_num += 1
    images0 = images0.cuda()
    labels0 = labels.cpu().data.numpy()
    # print labels0
    images = images0 * 1
    
    labels = labels.float().cuda()
    cover_mask = cover_net(images)
    class_out = class_net(images, cover_mask)
    print(class_out)
    class_out = torch.mean(class_out, dim=1)
    loss_class = (class_out - labels)
    loss_class = (loss_class * loss_class).mean()
    loss_cover = torch.mean(cover_mask)
    area_add += loss_cover.cpu().data.numpy()

    class_out = class_out.clamp(0, 5)
    error_b = (class_out - labels).abs()
    error_b += 0.5
    error_b = error_b.floor().sum().cpu().data.numpy()
    error_num += error_b

    loss_class_c = loss_class.cpu().data.numpy()
    loss_cover_c = loss_cover.cpu().data.numpy()

    print(" loss_class:", loss_class_c, "   loss_cover:", loss_cover_c)
    batch_max = images.size(0)
    for b in range(batch_max):
        image_one = images0[b:b + 1, :, :, :]
        cover_one = cover_mask[b:b + 1, :, :, :]
        # overlay_one = image_one*1
        # overlay_one[:,2,:,:] = overlay_one[:,2,:,:]*(1-cover_one)
        image_one = util.torch2numpy(image_one * 255)
        cover_one = np.squeeze(util.torch2numpy(cover_one * 255))
        # overlay_one = util.torch2numpy(overlay_one*255)
        overlay_one = image_one * 1
        overlay_one[:, :, 2] = cv2.add(overlay_one[:, :, 2], cover_one)
        save_path = out_path + str(labels0[b]) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path + str(save_num) + '_a.jpg', image_one)
        cv2.imwrite(save_path + str(save_num) + '_b.jpg', overlay_one)
        cv2.imwrite(save_path + str(save_num) + '_c.jpg', cover_one)
        save_num += 1

print("error:", float(error_num) / float(save_num))
print("area:", area_add / float(batch_num))
