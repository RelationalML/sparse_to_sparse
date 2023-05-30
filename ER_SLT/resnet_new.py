
from __future__ import print_function
import argparse
from enum import auto
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import operator
from functools import reduce
import numpy as np
import torch.nn.functional as F


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class GetSubnetER(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k, er_mask):
        # er_mask is the flattened initialized mask of the weight and bias combined as passed in the forward func
        retain_idx = torch.where(er_mask == 1)[0]
        out = scores.clone()        
        temp = scores[retain_idx]
        target_elem = int((k) * scores.numel())
        
        _, idx = temp.sort(descending=True)
        temp[idx[:target_elem]] = 1
        temp[idx[target_elem:]] = 0
        out[retain_idx] = temp

        return out
        
    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None


class SupermaskConv(nn.Conv2d):
    def __init__(self, sparsity, er_init_density, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scoresWeights = nn.Parameter(torch.Tensor(self.weight.size()))
        #nn.init.kaiming_uniform_(self.scoresWeights, a=math.sqrt(5))
        self.scoresBias = nn.Parameter(torch.Tensor(self.bias.size()))
        #fan = nn.init._calculate_correct_fan(self.scoresWeights, 'fan_in')
        #bound = math.sqrt(6.0/fan)
        #nn.init.uniform_(self.scoresBias, -bound, bound)
        #nn.init.kaiming_uniform_(self.scoresBias, a=math.sqrt(5))
        
        self.er_mask = torch.empty(self.weight.size(), device = self.weight.device).bernoulli_(p = er_init_density)
        nn.init.constant_(self.scoresBias,0.5)
        nn.init.constant_(self.scoresWeights, 0.5)

        self.sparsity = sparsity

        # NOTE: initialize the weights like this.
        #nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        subnet = GetSubnetER.apply(torch.cat((self.scoresWeights.abs().flatten(), self.scoresBias.abs().flatten())), self.sparsity, self.er_mask)
        w = self.weight * subnet[:self.scoresWeights.numel()].view(self.scoresWeights.size())
        b = self.bias * subnet[self.scoresWeights.numel():].view(self.scoresBias.size())
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SupermaskLinear(nn.Linear):
    def __init__(self, sparsity, er_init_density, *args, **kwargs):
    #def __init__(self, sparsity, zerobias, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scoresWeights = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scoresWeights, a=math.sqrt(5))
        self.scoresBias = nn.Parameter(torch.Tensor(self.bias.size()))
        fan = nn.init._calculate_correct_fan(self.scoresWeights, 'fan_in')
        

        self.er_mask = torch.empty(self.weight.size(), device = self.weight.device).bernoulli_(p = er_init_density)

        bound = math.sqrt(6.0/fan)
        nn.init.uniform_(self.scoresBias, -bound, bound)
        #nn.init.constant_(self.scoresBias,0.5)
        #nn.init.constant_(self.scoresWeights, 0.5)
        #nn.init.uniform_(self.scoresBias, 0.25, 0.75)
        #nn.init.uniform_(self.scoresWeights, 0.25, 0.75)
        #nn.init.kaiming_uniform_(self.scoresBias, a=math.sqrt(5))
        
        
        #self.zero_bias = zerobias

        self.sparsity = sparsity

        # NOTE: initialize the weights like this.
        #nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        #scw = self.scoresWeights.abs().flatten()
        #scb = self.scoresBias.abs().flatten()
        #factor=1
        #mw = torch.mean(scw[scw>0])
        #mb = torch.mean(scb[scb>0])
        #print("new")
        #print(mb)
        #print(mw)
        #if mb > 0.00001:
        #    factor = mw/mb
       #     factor = 1
       #     print("new")
       #     print(scb[:10])
       #     print(scw[:10])
        #else:
        #    factor = 1
        #print(factor)
        #print(self.bias.size())
        #if self.zero_bias:
        #    self.scoresBias.data = torch.zeros(self.scoresBias.size(0))
        subnet = GetSubnetER.apply(torch.cat((self.scoresWeights.abs().flatten(), self.scoresBias.abs().flatten())), self.sparsity, self.er_mask)
        #subnet = GetSubnet.apply(torch.cat((self.scoresWeights.abs().flatten(), factor*self.scoresBias.abs().flatten())), self.sparsity)
        #print(subnet[self.scoresWeights.numel():].view(self.scoresBias.size()).size())
        w = self.weight * subnet[:self.scoresWeights.numel()].view(self.scoresWeights.size())
        b = self.bias[:self.scoresBias.size(0)] * subnet[self.scoresWeights.numel():].view(self.scoresBias.size())
        #self.scoresBias.data = subnet[self.scoresWeights.numel():].view(self.scoresBias.size())
        #self.scoresWeights.data = subnet[:self.scoresWeights.numel()].view(self.scoresWeights.size())
        #print(subnet.size())
        #print(self.scoresWeights.numel())
        #
        #b = self.bias * subnet[self.scoresWeights.numel():].view(self.scoresBias.size())
        x = F.linear(x, w, b)
        return x #F.linear(x, w, b)
        #x = F.linear(x, w, b)
        #return x



def conv3x3(sparsity, er_init_density, in_planes, out_planes, stride=1):
    return SupermaskConv(sparsity, er_init_density, in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, sparsity, er_init_density, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(sparsity, er_init_density, in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.conv2 = conv3x3(sparsity, er_init_density, planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                SupermaskConv(sparsity, er_init_density, in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes, affine=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, sparsity, er_init_density, num_classes):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        block, num_blocks = (BasicBlock, [2,2,2,2])

        self.conv1 = conv3x3(sparsity, er_init_density, 3, 64)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.layer1 = self._make_layer(sparsity, er_init_density, block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(sparsity, er_init_density, block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(sparsity, er_init_density, block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(sparsity, er_init_density, block, 512, num_blocks[3], stride=2)
        self.linear = SupermaskLinear(sparsity, er_init_density, 512*block.expansion, num_classes)


    def _make_layer(self, sparsity, er_init_density, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(sparsity, er_init_density, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def clampScores(self, min=0):
        with torch.no_grad():
            l = [module for module in self.modules() if isinstance(module, (SupermaskConv, SupermaskLinear))]
            for layer in l:
                layer.scoresWeights.clamp_(min=min)
                layer.scoresBias.clamp_(min=min)

    # def _initialize_weights(self, initializer):
    #     for m in self.modules():
    #         if isinstance(m, (SupermaskLinear, SupermaskConv)):
    #             nn.init.kaiming_normal_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
