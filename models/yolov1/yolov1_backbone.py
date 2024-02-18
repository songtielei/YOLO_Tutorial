import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


# ResNet的ImageNet pretrained权重的链接
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# --------------------- 基础模块 -----------------------
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# --------------------- ResNet网络 -----------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, C, H, W]
        Output:
            c5: (Tensor) -> [B, C, H/32, W/32]
        """
        c1 = self.conv1(x)     # [B, C, H/2, W/2]
        c1 = self.bn1(c1)      # [B, C, H/2, W/2]
        c1 = self.relu(c1)     # [B, C, H/2, W/2]
        c2 = self.maxpool(c1)  # [B, C, H/4, W/4]

        c2 = self.layer1(c2)   # [B, C, H/4, W/4]
        c3 = self.layer2(c2)   # [B, C, H/8, W/8]
        c4 = self.layer3(c3)   # [B, C, H/16, W/16]
        c5 = self.layer4(c4)   # [B, C, H/32, W/32]

        return c5


# --------------------- 构建ResNet网络的函数 -----------------------
## 搭建ResNet-18网络
def resnet18(pretrained=False, **kwargs):
    """搭建 ResNet-18 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # strict = False as we don't need fc layer params.
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

## 搭建ResNet-34网络
def resnet34(pretrained=False, **kwargs):
    """搭建 ResNet-34 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model

## 搭建ResNet-50网络
def resnet50(pretrained=False, **kwargs):
    """搭建 ResNet-50 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

## 搭建ResNet-101网络
def resnet101(pretrained=False, **kwargs):
    """搭建 ResNet-101 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model

## 搭建ResNet-152网络
def resnet152(pretrained=False, **kwargs):
    """搭建 ResNet-152 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

## 搭建ResNet网络
def build_backbone(model_name='resnet18', pretrained=False):
    if model_name == 'resnet18':
        model = resnet18(pretrained)
        feat_dim = 512    # 网络的最终输出的feature的通道维度为512
    elif model_name == 'resnet34':
        model = resnet34(pretrained)
        feat_dim = 512    # 网络的最终输出的feature的通道维度为512
    elif model_name == 'resnet50':
        model = resnet34(pretrained)
        feat_dim = 2048   # 网络的最终输出的feature的通道维度为2048
    elif model_name == 'resnet101':
        model = resnet34(pretrained)
        feat_dim = 2048   # 网络的最终输出的feature的通道维度为2048

    return model, feat_dim


if __name__=='__main__':
    # 这是一段测试代码，方便读者测试能否正常的下载ResNet权重和调用ResNet网络
    model, feat_dim = build_backbone(model_name='resnet18', pretrained=True)

    # 打印模型的结构
    print(model)

    # 输入图像的参数
    batch_size    = 2
    image_channel = 3
    image_height  = 512
    image_width   = 512

    # 随机生成一张图像
    image = torch.randn(batch_size, image_channel, image_height, image_width)

    # 模型推理
    output = model(image)

    # 查看模型的输出的shape
    print(output.shape)


    