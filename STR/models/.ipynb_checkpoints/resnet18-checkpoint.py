'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.builder import get_builder
from args import args

class BasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.relu = builder.activation()
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = builder.batchnorm(planes, last_bn=True)
        downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            dconv = builder.conv1x1(
                inplanes, planes * self.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * self.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out 



class Bottleneck(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = builder.conv1x1(inplanes, planes)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=stride)
        self.bn2 = builder.batchnorm(planes)
        self.conv3 = builder.conv1x1(planes, planes * self.expansion)
        self.bn3 = builder.batchnorm(planes * self.expansion, last_bn=True)
        self.relu = builder.activation()

        downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            dconv = builder.conv1x1(
                inplanes, planes * self.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * self.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, builder, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = builder.conv3x3(3, 64)
        
        self.bn1 = builder.batchnorm(64)
        self.layer1 = self._make_layer(builder, block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(builder, block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(builder, block, 512, num_blocks[3], stride=2)
        self.linear = builder.conv1x1(512*block.expansion, num_classes)

    def _make_layer(self, builder, block, planes, num_blocks, stride):
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = self.linear(out).squeeze()

        return out


def ResNet18(input_shape, num_classes, dense_classifier=False, pretrained=True):
    return ResNet(get_builder(), BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(input_shape, num_classes, dense_classifier=False, pretrained=True):
    return ResNet(get_builder(), BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(input_shape, num_classes, dense_classifier=False, pretrained=True):
    return ResNet(get_builder(), Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(input_shape, num_classes, dense_classifier=False, pretrained=True):
    return ResNet(get_builder(), Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(input_shape, num_classes, dense_classifier=False, pretrained=True):
    return ResNet(get_builder(), Bottleneck, [3, 8, 36, 3], num_classes)




# test()


class ResNetWidth(nn.Module):
    def __init__(self, builder, block, num_blocks, width, num_classes=10):
        super(ResNetWidth, self).__init__()
        self.in_planes = 64

        self.conv1 = builder.conv3x3(3, 64)
        
        self.bn1 = builder.batchnorm(64)
        self.layer1 = self._make_layer(builder, block, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(builder, block, width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(builder, block, width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(builder, block, width, num_blocks[3], stride=2)
        self.linear = builder.conv1x1(width*block.expansion, num_classes)

    def _make_layer(self, builder, block, planes, num_blocks, stride):
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = self.linear(out).squeeze()

        return out


def ResNetWidth18(input_shape, num_classes, width=64, dense_classifier=False, pretrained=True):
    return ResNetWidth(get_builder(), BasicBlock, [2, 2, 2, 2], width, num_classes)


def ResNetWidth34(input_shape, num_classes, width=128, dense_classifier=False, pretrained=True):
    return ResNetWidth(get_builder(), BasicBlock, [3, 4, 6, 3], width, num_classes)


def ResNetWidth50(input_shape, num_classes, width=128, dense_classifier=False, pretrained=True):
    return ResNetWidth(get_builder(), Bottleneck, [3, 4, 6, 3], width, num_classes)
