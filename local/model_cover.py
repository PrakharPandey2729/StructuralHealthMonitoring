# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import resize


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
def rp(x):
    out = F.pad(x, (1,1,1,1), "replicate")
    return out

def adjust_num(input0, output_num):

    input_num = input0.size(1)
    rest = output_num%input_num
    cop = output_num/input_num
    if rest==0 and cop==1:
        return input0
    elif rest:
        con_list = [input0[:,0:rest,:,:]]
    else:
        con_list = []
    for i in range(cop):
        con_list.append(input0)
    out_put = torch.cat(con_list,dim=1)
    return out_put

def rp(x):
    out = F.pad(x, (1,1,1,1), "replicate")
    return out

class COVER_NET(nn.Module):

    def __init__(self):
        super(COVER_NET, self).__init__()

        self.avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        self.mrelu = nn.ReLU(inplace=False)
        # self.lrelu = nn.LeakyReLU(inplace=False)
        self.lrelu = nn.LeakyReLU(inplace=True)

        self.grid_net_init('cover',height=5,width=6,channel=[32, 64, 96, 128, 196],in_c=3,out_c=1)



    def grid_net_init(self,name,height,width,channel,in_c,out_c):
        setattr(self,name+'_'+'input',
                nn.Conv2d(in_channels=in_c, out_channels=channel[0],
                          kernel_size=3,stride=1, padding=0))
        setattr(self,name+'_'+'output',
                nn.Conv2d(in_channels=channel[0], out_channels=out_c,
                          kernel_size=3,stride=1, padding=0))
        self.height = height
        self.width = width
        momentum = 0.05
        for i in range(height):
            for j in range(width):
                # 向右
                if j+1 < width:
                    setattr(self, name + '_conv0_' + str(i) + '_'+ str(j) + '_'+ str(i) + '_'+ str(j+1),
                            nn.Conv2d(in_channels=channel[i], out_channels=channel[i],
                                      kernel_size=3, stride=1, padding=0))
                    setattr(self, name + '_norm0_' + str(i) + '_'+ str(j) + '_'+ str(i) + '_'+ str(j+1),
                            nn.BatchNorm2d(channel[i], momentum=momentum))

                # 向下
                if i+1 < height and j < width/2:
                    setattr(self, name + '_conv0_' + str(i) + '_'+ str(j) + '_'+ str(i+1) + '_'+ str(j),
                            nn.Conv2d(in_channels=channel[i], out_channels=channel[i+1],
                                      kernel_size=3, stride=2, padding=0))
                    setattr(self, name + '_norm0_' + str(i) + '_' + str(j) + '_' + str(i + 1) + '_' + str(j),
                            nn.BatchNorm2d(channel[i], momentum=momentum))
                # 向上
                if i - 1 >= 0 and j >= width / 2:
                    setattr(self, name + '_conv0_' + str(i) + '_' + str(j) + '_' + str(i - 1) + '_' + str(j),
                            nn.Conv2d(in_channels=channel[i], out_channels=channel[i-1],
                                      kernel_size=3, stride=1, padding=0))
                    setattr(self, name + '_norm0_' + str(i) + '_' + str(j) + '_' + str(i - 1) + '_' + str(j),
                            nn.BatchNorm2d(channel[i], momentum=momentum))
        pass

    def grid_net_forward(self,input0,name):
        conv_out = []
        for i in range(self.height):
            conv_out.append([])
            for j in range(0,self.width):
                conv_out[i].append(0)
        conv_out[0][0] = self.lrelu(getattr(self,name+'_'+'input')(rp(input0)))

        for i in range(self.height):
            for j in range(0, self.width // 2):
                # 来自左
                if j - 1 >= 0:
                    input_0 = conv_out[i][j - 1]
                    input_1 = getattr(self, name + '_norm0_' + str(i) + '_' +
                                      str(j - 1) + '_' + str(i) + '_' + str(j))(input_0)
                    input_1 = rp(self.lrelu(input_1))
                    conv_out[i][j] = conv_out[i][j] +\
                        getattr(self, name + '_conv0_' + str(i) + '_' +
                                str(j - 1) + '_' + str(i) + '_' + str(j))(input_1) + input_0
                # 来自上
                if i - 1 >= 0:
                    input_0 = conv_out[i - 1][j]
                    input_1 = getattr(self, name + '_norm0_' + str(i - 1) + '_' +
                                      str(j) + '_' + str(i) + '_' + str(j))(input_0)
                    input_1 = rp(self.lrelu(input_1))
                    input_1 = getattr(self, name + '_conv0_' + str(i - 1) + '_' + str(j) +
                                      '_' + str(i) + '_' + str(j))(input_1)

                    conv_out[i][j] = conv_out[i][j] + input_1
        for i in range(self.height - 1, -1, -1):
            for j in range(self.width // 2, self.width):
                # 来自左
                input_0 = conv_out[i][j - 1]
                input_1 = getattr(self, name + '_norm0_' + str(i) + '_' +
                                  str(j - 1) + '_' + str(i) + '_' + str(j))(input_0)
                input_1 = rp(self.lrelu(input_1))
                conv_out[i][j] = conv_out[i][j] + \
                                 getattr(self, name + '_conv0_' + str(i) + '_' +
                                         str(j - 1) + '_' + str(i) + '_' + str(j))(input_1) + input_0

                # 来自下
                if i + 1 < self.height:
                    h = conv_out[i][j].size(2)
                    w = conv_out[i][j].size(3)
                    re_input = resize(conv_out[i + 1][j], [h, w])
                    re_input = getattr(self, name + '_norm0_' + str(i + 1) + '_' +
                                      str(j) + '_' + str(i) + '_' + str(j))(re_input)
                    re_input = rp(self.lrelu(re_input))
                    conv = getattr(self, name + '_conv0_' + str(i + 1) + '_' + str(j) +
                                   '_' + str(i) + '_' + str(j))(re_input)

                    conv_out[i][j] += conv

        output0 = getattr(self,name+'_'+'output')(rp(conv_out[0][self.width-1]))

        return output0

    def forward(self, img_input):
        img_input = img_input *1.0


        cover_out = self.grid_net_forward(img_input,'cover')

        cover_out = F.softplus(cover_out+20)


        b,c,h,w = cover_out.size()
        cover_max = F.max_pool2d(cover_out,(h,w),(h,w),padding=0).expand(b, c, h, w)
        cover_out = cover_out/(cover_max+1e-8)


        return cover_out


