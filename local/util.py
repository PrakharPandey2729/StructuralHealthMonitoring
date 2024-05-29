# coding=utf-8
import torch
import torch.nn.functional as F
import numpy as np
import os

def numpy2torch(npy):
    npy0 = npy[np.newaxis, :, :, :]
    ter = torch.from_numpy(npy0)
    ter = ter.cuda()
    ter = torch.transpose(ter, dim0=1, dim1=3)
    ter = torch.transpose(ter, dim0=2, dim1=3).float()
    return ter

def torch2numpy(ter):
    ter = torch.squeeze(ter, dim=0)
    ter = torch.transpose(ter, dim0=0, dim1=2)
    ter = torch.transpose(ter, dim0=0, dim1=1)
    npy = ter.cpu().data.numpy()
    return npy

def models_parameters(model_list):
    for model in model_list:
        for name, param in model.named_parameters():
            yield param


def b_warp(pic_in, flow):
    h = pic_in.size(2)
    w = pic_in.size(3)
    b = pic_in.size(0)

    flow0 = torch.zeros_like(flow)
    flow0[:, 0, :, :] = flow[:, 0, :, :] / w * 2
    flow0[:, 1, :, :] = flow[:, 1, :, :] / h * 2

    torchHorizontal = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(b, 1, h, w)
    torchVertical = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(b, 1, h, w)
    grid = torch.cat([torchHorizontal, torchVertical], 1).cuda()
    pic_in = pic_in.float()
    pic_out = F.grid_sample(pic_in, (grid + flow0).permute(0, 2, 3, 1),padding_mode='border')
    return pic_out

def resize(input,size_out):
    out_put = F.interpolate(input,size_out,mode='bilinear')
    return out_put

def plt_change(img_in):
    img_out = np.zeros_like(img_in)
    img_out[:, :, 0] = img_in[:, :, 2] * 1
    img_out[:, :, 1] = img_in[:, :, 1] * 1
    img_out[:, :, 2] = img_in[:, :, 0] * 1
    return img_out

def save_weight(model,step,path):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = path + "ckpt_" + str(step) + ".pkl"
    if torch.cuda.is_available():
        state = model.state_dict()
        torch.save(state, save_path)
    else:
        torch.save(model.state_dict(), save_path)
    step_path = path + "step.txt"
    with file(step_path,'w') as f:
        f.write(str(step))
        f.close()

def adjust_num(input, output_num):

    input_num = input.size(1)
    rest = output_num%input_num
    cop = output_num/input_num
    if rest:
        con_list = [input[:,0:rest,:,:]]
    else:
        con_list = []
    for i in range(cop):
        con_list.append(input)
    out_put = torch.cat(con_list,dim=1)
    return out_put

def get_parameters(key_word_list,model):
    for name, param in model.named_parameters():
        for key_word in key_word_list:
            if key_word in name:
                yield param
                break

def load_weight(path):
    if not os.path.exists(path):
        os.makedirs(path)
    step_path = path + "step.txt"
    if os.path.exists(step_path):
        with file(step_path, 'r') as f:
            step = f.read()
            place = step.find('\n')
            if place != -1:
                step = step[:place]
            f.close()
        save_path = path + "ckpt_" + step + ".pkl"
        # save_path = './ckpt/inter_2000.pkl'
        print("read weight:",save_path)
        return save_path,int(step)
    else:
        print("new model:")
        return None,0
