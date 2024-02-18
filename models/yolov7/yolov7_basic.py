import numpy as np
import torch
import torch.nn as nn


# ----------------------- 常用的基础模块 -----------------------
class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_conv2d(c1, c2, k, p, s, d, g, bias=False):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)

    return conv

def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    elif act_type is not None:
        return nn.Identity()
    else:
        raise NotImplementedError('Activation {} not implemented.'.format(act_type))

def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is not None:
        return nn.Identity()
    else:
        raise NotImplementedError('Normalization {} not implemented.'.format(norm_type))

class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # 输入通道数
                 c2,                   # 输出通道数 
                 k=1,                  # 卷积核尺寸 
                 p=0,                  # 补零的尺寸
                 s=1,                  # 卷积的步长
                 d=1,                  # 卷积膨胀系数
                 act_type='lrelu',     # 激活函数的类别
                 norm_type='BN',       # 归一化层的类别
                 depthwise=False       # 是否使用depthwise卷积
                 ):
        super(Conv, self).__init__()
        convs = []
        add_bias = False if norm_type else True

        # 构建depthwise + pointwise卷积
        if depthwise:
            convs.append(get_conv2d(c1, c1, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            # 首先，搭建depthwise卷积
            if norm_type:
                convs.append(get_norm(norm_type, c1))
            if act_type:
                convs.append(get_activation(act_type))
            # 然后，搭建pointwise卷积
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))

        # 构建普通的标准卷积
        else:
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


# ----------------------- YOLOv7的必要组件 -----------------------
## ELAN-Block proposed by YOLOv7
class ELANBlock(nn.Module):
    def __init__(self, in_dim, out_dim, squeeze_ratio=0.5, branch_depth :int=2, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANBlock, self).__init__()
        inter_dim = int(in_dim * squeeze_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv3 = nn.Sequential(*[
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(round(branch_depth))
        ])
        self.cv4 = nn.Sequential(*[
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            for _ in range(round(branch_depth))
        ])

        self.out = Conv(inter_dim*4, out_dim, k=1, act_type=act_type, norm_type=norm_type)



    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

        return out

## PaFPN's ELAN-Block proposed by YOLOv7
class ELANBlockFPN(nn.Module):
    def __init__(self, in_dim, out_dim, squeeze_ratio=0.5, branch_width :int=4, branch_depth :int=1, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANBlockFPN, self).__init__()
        # Basic parameters
        inter_dim = int(in_dim * squeeze_ratio)
        inter_dim2 = int(inter_dim * squeeze_ratio) 
        # Network structure
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv3 = nn.ModuleList()
        for idx in range(round(branch_width)):
            if idx == 0:
                cvs = [Conv(inter_dim, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            else:
                cvs = [Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            # deeper
            if round(branch_depth) > 1:
                for _ in range(1, round(branch_depth)):
                    cvs.append(Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise))
                self.cv3.append(nn.Sequential(*cvs))
            else:
                self.cv3.append(cvs[0])

        self.out = Conv(inter_dim*2+inter_dim2*len(self.cv3), out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        inter_outs = [x1, x2]
        for m in self.cv3:
            y1 = inter_outs[-1]
            y2 = m(y1)
            inter_outs.append(y2)
        out = self.out(torch.cat(inter_outs, dim=1))

        return out

## DownSample Block proposed by YOLOv7
class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        inter_dim = out_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)
        out = torch.cat([x1, x2], dim=1)

        return out
