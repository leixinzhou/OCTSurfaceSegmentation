from __future__ import print_function
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1
    )

def conv1(in_channels, out_channels):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=1
    )


class DsBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pooling):
        super(DsBlock, self).__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = pooling
        if pooling:
            self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):

        out = self.conv(x)
        out = self.relu(out)
        before_pool = out
        if self.pooling:
            out = self.mp(out)

        return out, before_pool


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


class UsBlock(nn.Module):

    def __init__(self, in_channels, out_channels, up_mode='transpose'):
        super(UsBlock, self).__init__()

        self.upconv = upconv2x2(in_channels, out_channels, mode=up_mode)
        self.conv = conv3x3(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, before_pool, x):
        x = self.upconv(x)
        x = x + before_pool
        x = self.conv(x)
        x = self.relu(x)
        return x


class FCN(nn.Module):
    '''
    This class implements a Unet with global and local residual connections. The input construction arguments: num_classes, 
    in_channels, depth (down sampling number is depth-1), start_filter number, upsampling mode.
    '''

    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose'):
        super(FCN, self).__init__()

        self.down_convs = []
        self.up_convs = []

        # put one conv  at the beginning
        self.conv_start = conv3x3(in_channels, start_filts, stride=1)
        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = start_filts * (2 ** i)
            outs = start_filts * (2 ** (i + 1))
            pooling = True if i < depth - 1 else False

            down_conv = DsBlock(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UsBlock(ins, outs, up_mode=up_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, x):
        x = self.conv_start(x)
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        # print(len(encoder_outs))
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        return x

class FCN_D(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose', col_len=512, fc_inter=128):
        super(FCN_D, self).__init__()
        self.FCN = FCN(num_classes, in_channels, depth,
                 start_filts, up_mode)
        # self.FC_G1 = conv1(col_len, fc_inter)
        # self.FC_G2 = conv1(fc_inter, 1)
        self.FC_D1 = conv1(col_len, fc_inter)
        self.FC_D2 = conv1(fc_inter, 1)

    def forward(self, x):
        x = self.FCN(x).squeeze(1)
        x = x.permute(0,2,1)
        # G = self.FC_G1(x)
        # G = torch.nn.functional.relu(G)
        # G = self.FC_G2(G)
        D = self.FC_D1(x[:,:,:-1] - x[:,:,1:])
        D = torch.nn.functional.relu(D)
        D = self.FC_D2(D)
        return D

class FCN_D2(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose', col_len=512, fc_inter=128, left_nbs=3, **kwargs):
        super(FCN_D2, self).__init__()
        self.FCN = FCN(num_classes, in_channels, depth,
                 start_filts, up_mode)
        # self.FC_G1 = conv1(col_len, fc_inter)
        # self.FC_G2 = conv1(fc_inter, 1)
        self.FC_D1 = conv1(col_len*(left_nbs+1)*2, fc_inter)
        self.FC_D2 = conv1(fc_inter, 1)
        self.half = left_nbs+1

    def forward(self, x):
        x = self.FCN(x).squeeze(1)
        # x = x.permute(0,2,1)
        # G = self.FC_G1(x)
        # G = torch.nn.functional.relu(G)
        # G = self.FC_G2(G)
        t_list = []
        for i in range(self.half):
            if i==0:
                t_list.append(x[:,:,:-1])
                t_list.append(x[:,:,1:])
            else:
                t_list.append(torch.cat((x[:,:,0:1].expand(-1,-1,i), x[:,:,:-(i+1)]), -1))
                t_list.append(torch.cat((x[:,:,i+1:], x[:,:,-1:].expand(-1,-1,i)), -1))
        t_list = torch.cat(t_list, 1)
        # print(t_list.size())
        D = self.FC_D1(t_list)
        D = torch.nn.functional.relu(D)
        D = self.FC_D2(D)
        D = D.squeeze(1)
        return D
class ConvNet(nn.Module):
    def __init__(self, in_channels=1, depth=5,
                 start_filts=32):
        super(ConvNet, self).__init__()

        self.down_convs = []

        # put one conv  at the beginning
        self.conv_start = conv3x3(in_channels, start_filts, stride=1)
        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = start_filts * (2 ** i)
            outs = start_filts * (2 ** (i + 1))
            pooling = True 

            down_conv = DsBlock(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)

    def forward(self, x):
        x = self.conv_start(x)

        for i, module in enumerate(self.down_convs):
            x, _ = module(x)

        return x

class ALEX(nn.Module):
    def __init__(self, in_channels=1, depth=5, start_filters=32, fc_inter=512, col_nb=400, col_len=512, **kwargs):
        super(ALEX, self).__init__()
        self.convnet = ConvNet(in_channels=in_channels, depth=depth, start_filts=start_filters)
        self.fc_input = int((col_nb/2**depth) * col_len * start_filters)
        self.FC_D1 = nn.Linear(self.fc_input, fc_inter)
        self.FC_D2 = nn.Linear(fc_inter, col_nb-1)
    def forward(self, x):
        x = self.convnet(x)
        # print(self.fc_input, x.size())
        x = x.view(x.size(0), self.fc_input)
        x = self.FC_D1(x)
        x = nn.functional.relu(x)
        x = self.FC_D2(x)
        return x

if __name__ == "__main__":
    """
    testing
    """
    model = FCN_FC(num_classes=1, in_channels=1, depth=3,
                start_filts=8, up_mode='transpose', col_len=32, fc_inter=8)
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(p_count)
    x = Variable(torch.FloatTensor(np.random.random((10, 1, 16, 32))))
    y = Variable(torch.FloatTensor(np.random.uniform(0,31,(10,16))))
    G, _ = model(x)
    loss_func = torch.nn.L1Loss()
    loss = loss_func(G[:,0], y)
    loss.backward()
