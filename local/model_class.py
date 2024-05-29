import torch
import torch.nn as nn
import torch.nn.functional as F
from util import resize

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(outchannel,momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel,momentum=0.05)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(outchannel,momentum=0.05)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64,momentum=0.05),
            nn.ReLU(),
        )
        self.mid = False
        self.layer1 = self.make_layer(ResidualBlock, 32,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 96, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer5 = self.make_layer(ResidualBlock, 196, 2, stride=2)
        self.fc0 = nn.Linear(50176, 64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x, cover_mask):
        if self.mid:
            x,mid_list = self.cover(x,cover_mask,self.mid)
        else:
            x = self.cover(x, cover_mask)
        out = self.conv1(x)
        out = self.cover(out, cover_mask)
        out = self.layer1(out)
        out = self.cover(out, cover_mask)
        out = self.layer2(out)
        out = self.cover(out, cover_mask)
        out = self.layer3(out)
        # out = self.cover(out, cover_mask)
        out = self.layer4(out)
        # out = self.cover(out, cover_mask)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        # print out.size()
        out = F.relu(self.fc0(out))
        out = F.relu(self.fc1(out)+out)
        out = F.relu(self.fc2(out)+out)
        logits = self.fc3(out)
        # probabilities = F.softmax(logits, dim=1)

        if self.mid:
            return logits,mid_list
        else:
            return logits

    def cover(self,fea,cover_mask,mid=False):
        if mid:
            mid_list = []
        b,c,h,w = fea.size()
        cover_resize = resize(cover_mask,[h,w]).expand(b, c, h, w)
        fea_max = F.max_pool2d(fea, (h, w), (h, w), padding=0).expand(b, c, h, w)
        fea_min = -F.max_pool2d(-fea, (h, w), (h, w), padding=0).expand(b, c, h, w)
        fea = (fea-fea_min)/(fea_max-fea_min+1e-8)
        if mid:
            mid_list.append(fea*1)
        noise = torch.rand(fea.size()).cuda()
        if mid:
            mid_list.append(noise*1)
            mid_list.append(noise*(1-cover_resize))
        fea = fea*cover_resize+noise*(1-cover_resize)
        if mid:
            mid_list.append(fea*1)
        fea = fea*(fea_max-fea_min+1e-8)+fea_min

        if mid:
            mid_list.append(fea*1)
            return fea,mid_list
        else:
            return fea

    def mid_on(self):
        self.mid = False




def CLASS_NET():

    return ResNet(ResidualBlock)
















