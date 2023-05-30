import torch
import torch.nn as nn
import math
import numpy as np

import torch.optim

import resnet_new


def init_with_bias(args, model):
    gainProd = args.bfac #0.05 #0.06 #0.07 #0.06 #0.09 #0.1 #0.03 #0.1 #1.0 #0.5
    for m in model.modules():
        if isinstance(m,(resnet_new.SupermaskConv, resnet_new.SupermaskLinear, nn.Linear, nn.Conv2d)):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2/fan_in)
            if args.scale_fan:
                std = std / math.sqrt(args.sparsity) #math.sqrt(1-args.prune_rate) #* (1 - args.prune_rate) #/ math.sqrt(args.prune_rate) #(1 - args.prune_rate)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            m.weight.requires_grad = False
            gainProd = gainProd*std
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=gainProd)
                m.bias.requires_grad = False

def signed_const_zero_bias(args, model):
    gainProd = args.bfac #0.05 #0.06 #0.07 #0.06 #0.09 #0.1 #0.03 #0.1 #1.0 #0.5
    for m in model.modules():
        if isinstance(m,(resnet_new.SupermaskConv, resnet_new.SupermaskLinear, nn.Linear, nn.Conv2d)):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2/fan_in)
            if args.scale_fan:
                std = std / math.sqrt(args.sparsity) #math.sqrt(1-args.prune_rate) #* (1 - args.prune_rate) #/ math.sqrt(args.prune_rate) #(1 - args.prune_rate)
            nn.init.uniform_(m.weight, 0, 1)
            m.weight.data = std * torch.where(m.weight >= 0.5, 1, -1)
            
            m.weight.requires_grad = False
            gainProd = gainProd*std
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                m.bias.requires_grad = False
    

def signed_const_with_bias(args, model):
    gainProd = args.bfac #0.05 #0.06 #0.07 #0.06 #0.09 #0.1 #0.03 #0.1 #1.0 #0.5
    for m in model.modules():
        if isinstance(m,(resnet_new.SupermaskConv, resnet_new.SupermaskLinear, nn.Linear, nn.Conv2d)):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2/fan_in)
            if args.scale_fan:
                std = std / math.sqrt(args.sparsity) #math.sqrt(1-args.prune_rate) #* (1 - args.prune_rate) #/ math.sqrt(args.prune_rate) #(1 - args.prune_rate)
            nn.init.uniform_(m.weight, 0, 1)
            m.weight.data = std * torch.where(m.weight >= 0.5, 1, -1)
            
            m.weight.requires_grad = False
            gainProd = gainProd*std
            if m.bias is not None:
                nn.init.uniform_(m.bias, 0, 1)
                m.bias.data = std * torch.where(m.bias >= 0.5, 1, -1)
                m.bias.requires_grad = False

def init_with_uniform_bias_unscaled(args, model):
    gainProd = args.bfac #0.05 #0.06 #0.07 #0.06 #0.09 #0.1 #0.03 #0.1 #1.0 #0.5
    for m in model.modules():
        if isinstance(m,(resnet_new.SupermaskConv, resnet_new.SupermaskLinear, nn.Linear, nn.Conv2d)):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2/fan_in)
            if args.scale_fan:
                std = std / math.sqrt(args.sparsity) #math.sqrt(1-args.prune_rate) #* (1 - args.prune_rate) #/ math.sqrt(args.prune_rate) #(1 - args.prune_rate)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            m.weight.requires_grad = False
            gainProd = math.sqrt(1/fan_in)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -gainProd, gainProd)
                m.bias.requires_grad = False

def init_with_uniform_bias(args, model):
    gainProd = args.bfac #0.05 #0.06 #0.07 #0.06 #0.09 #0.1 #0.03 #0.1 #1.0 #0.5
    for m in model.modules():
        if isinstance(m,(resnet_new.SupermaskConv, resnet_new.SupermaskLinear, nn.Linear, nn.Conv2d)):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2/fan_in)
            if args.scale_fan:
                std = std / math.sqrt(args.sparsity) #math.sqrt(1-args.prune_rate) #* (1 - args.prune_rate) #/ math.sqrt(args.prune_rate) #(1 - args.prune_rate)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            m.weight.requires_grad = False
            gainProd = gainProd*std
            if m.bias is not None:
                nn.init.uniform_(m.bias, -gainProd, gainProd)
                m.bias.requires_grad = False


def init_zero_bias(args, model):
    gainProd = args.bfac #0.05 #0.06 #0.07 #0.06 #0.09 #0.1 #0.03 #0.1 #1.0 #0.5
    for m in model.modules():
        if isinstance(m,(resnet_new.SupermaskConv, resnet_new.SupermaskLinear, nn.Linear, nn.Conv2d)):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2/fan_in)
            if args.scale_fan:
                std = std / math.sqrt(args.sparsity) #math.sqrt(1-args.prune_rate) #* (1 - args.prune_rate) #/ math.sqrt(args.prune_rate) #(1 - args.prune_rate)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            m.weight.requires_grad = False
            gainProd = gainProd*std
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                m.bias.requires_grad = False
    
            
def init_with_bias_ortho(args, model):
    gainProd = args.bfac #0.05 #0.06
    i = 0
    for m in model.modules():
        if isinstance(m,(resnet_new.SupermaskConv, resnet_new.SupermaskLinear, nn.Linear, nn.Conv2d)):
            i=i+1
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2/fan_in)
            if i==1:
                #din = fan_in
                #dout = math.ceil(fan_out/2)
                dout = math.ceil(m.weight.size()[0]/2)
                din = m.weight.size()[1]
                #dbias = math.ceil(m.bias.size()/2) 
                if isinstance(m,(resnet_new.SupermaskConv)):
                    ww = nn.init.orthogonal_(torch.empty(dout, din, m.weight.size()[2], m.weight.size()[3], device=m.weight.device))
                    ww = torch.cat((ww,-ww),dim=0)
                    m.weight.data = ww[:m.weight.size()[0],:,:,:]
                else:
                    ww = nn.init.orthogonal_(torch.empty(dout, din, device=m.weight.device))
                    ww = torch.cat((ww,-ww),dim=0)
                    m.weight.data = ww[:m.weight.size()[0],:]
            else:
                din = math.ceil(m.weight.size()[1]/2)
                dout = math.ceil(m.weight.size()[0]/2)
                #dbias = math.ceil(m.bias.size()/2)
                if isinstance(m,(resnet_new.SupermaskConv)):
                    ww = nn.init.orthogonal_(torch.empty(dout, din, m.weight.size()[2], m.weight.size()[3], device=m.weight.device))
                    ww = torch.cat((ww,-ww),dim=0)
                    ww = torch.cat((ww,-ww),dim=1)
                    m.weight.data = ww[:m.weight.size()[0],:m.weight.size()[1],:,:]
                else:
                    ww = nn.init.orthogonal_(torch.empty(dout, din, device=m.weight.device))
                    ww = torch.cat((ww,-ww),dim=0)
                    ww = torch.cat((ww,-ww),dim=1)
                    m.weight.data = ww[:m.weight.size()[0],:m.weight.size()[1]]
            #also identify last layer?
            if args.scale_fan:
                std = std / math.sqrt(args.prune_rate)
            #nn.init.normal_(m.weight, mean=0.0, std=std)
            #dbias = math.ceil(m.bias.size()/2) 
            gainProd = gainProd*std
            if m.bias is not None:
                bb = torch.empty(dout, device=m.bias.device)
                nn.init.normal_(bb, mean=0.0, std=gainProd)
                bb = torch.cat((bb,-bb))
                m.bias.data = bb
            m.weight.requires_grad = False
            m.bias.requires_grad = False
                #nn.init.normal_(m.bias, mean=0.0, std=gainProd)
    # for n, m in model.named_modules():
    #     if isinstance(m, FixedSubnetConv):
    #         m.set_subnet()

def init_ortho_with_zero_bias(args, model):
    gainProd = args.bfac #0.05 #0.06
    i = 0
    for m in model.modules():
        if isinstance(m,(resnet_new.SupermaskConv, resnet_new.SupermaskLinear, nn.Linear, nn.Conv2d)):
            i=i+1
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2/fan_in)
            if i==1:
                #din = fan_in
                #dout = math.ceil(fan_out/2)
                dout = math.ceil(m.weight.size()[0]/2)
                din = m.weight.size()[1]
                #dbias = math.ceil(m.bias.size()/2)
                if isinstance(m,(resnet_new.SupermaskConv)):
                    ww = nn.init.orthogonal_(torch.empty(dout, din, m.weight.size()[2], m.weight.size()[3],device=m.weight.device))
                    ww = torch.cat((ww,-ww),dim=0)
                    m.weight.data = ww[:m.weight.size()[0],:,:,:]
                else:
                    ww = nn.init.orthogonal_(torch.empty(dout, din,device=m.weight.device))
                    ww = torch.cat((ww,-ww),dim=0)
                    m.weight.data = ww[:m.weight.size()[0],:]
            else:
                din = math.ceil(m.weight.size()[1]/2)
                dout = math.ceil(m.weight.size()[0]/2)
                #dbias = math.ceil(m.bias.size()/2)
                if isinstance(m,(resnet_new.SupermaskConv)):
                    ww = nn.init.orthogonal_(torch.empty(dout, din, m.weight.size()[2], m.weight.size()[3],device=m.weight.device))
                    ww = torch.cat((ww,-ww),dim=0)
                    ww = torch.cat((ww,-ww),dim=1)
                    m.weight.data = ww[:m.weight.size()[0],:m.weight.size()[1],:,:]
                else:
                    ww = nn.init.orthogonal_(torch.empty(dout, din,device=m.weight.device))
                    ww = torch.cat((ww,-ww),dim=0)
                    ww = torch.cat((ww,-ww),dim=1)
                    m.weight.data = ww[:m.weight.size()[0],:m.weight.size()[1]]
            #also identify last layer?
            if args.scale_fan:
                std = std / math.sqrt(args.prune_rate)
                m.weight.data =  m.weight.data / math.sqrt(args.prune_rate)
            #nn.init.normal_(m.weight, mean=0.0, std=std)
            #dbias = math.ceil(m.bias.size()/2) 
            gainProd = gainProd*std
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            m.weight.requires_grad = False
            m.bias.requires_grad = False
    # for n, m in model.named_modules():
    #     if isinstance(m, FixedSubnetConv):
    #         m.set_subnet()

def init_ortho_with_dep_bias(args, model):
    gainProd = args.bfac #0.05 #0.06
    i = 0
    for m in model.modules():
        if isinstance(m,(resnet_new.SupermaskConv, resnet_new.SupermaskLinear, nn.Linear, nn.Conv2d)):
            i=i+1
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2/fan_in)
            if i==1:
                dout = math.ceil(m.weight.size()[0]/2)
                din = m.weight.size()[1]
                #dbias = math.ceil(m.bias.size()/2) 
                if isinstance(m,(resnet_new.SupermaskConv,nn.Conv2d)):
                    ww = nn.init.orthogonal_(torch.empty(dout, din, m.weight.size()[2], m.weight.size()[3],device=m.weight.device))
                    ww = torch.cat((ww,-ww),dim=0)
                    m.weight.data = ww[:m.weight.size()[0],:,:,:]
                else:
                    ww = nn.init.orthogonal_(torch.empty(dout, din,device=m.weight.device))
                    ww = torch.cat((ww,-ww),dim=0)
                    m.weight.data = ww[:m.weight.size()[0],:]
            else:
                din = math.ceil(m.weight.size()[1]/2)
                dout = math.ceil(m.weight.size()[0]/2)
                #dbias = math.ceil(m.bias.size()/2)
                if isinstance(m,(resnet_new.SupermaskConv, nn.Conv2d)):
                    ww = nn.init.orthogonal_(torch.empty(dout, din, m.weight.size()[2], m.weight.size()[3],device=m.weight.device))
                    wprev = ww
                else:
                    ww = nn.init.orthogonal_(torch.empty(dout, din,device=m.weight.device))
                    wprev = ww
                if args.scale_fan:
                    wprev = wprev / math.sqrt(args.prune_rate)
                ww = torch.cat((ww,-ww),dim=0)
                ww = torch.cat((ww,-ww),dim=1)
                if isinstance(m,(resnet_new.SupermaskConv)):
                    m.weight.data = ww[:m.weight.size()[0],:m.weight.size()[1],:,:]
                else:
                    m.weight.data = ww[:m.weight.size()[0],:m.weight.size()[1]]
            #also identify last layer?
            if args.scale_fan:
                std = std / math.sqrt(args.prune_rate)
                m.weight.data = m.weight.data / math.sqrt(args.prune_rate)
            gainProd = gainProd*std
            if m.bias is not None:
                bb = torch.empty(dout,device=m.bias.device)
                nn.init.normal_(bb, mean=0.0, std=gainProd)
                if i%2 == 1:
                    bprev = bb
                    bb = torch.cat((bb,-bb))
                    m.bias.data = bb[:m.bias.size(0)]
                else:
                    scale = 1/math.sqrt(wprev.size(2)*wprev.size(3))
                    wprev = torch.sum(wprev, dim=(2,3))
                    bb = torch.matmul(wprev,bprev)*scale
                    bb = torch.cat((-bb,bb))
                    m.bias.data = bb[:m.bias.size(0)]
            m.weight.requires_grad = False
            m.bias.requires_grad = False
    # for n, m in model.named_modules():
    #     if isinstance(m, FixedSubnetConv):
    #         m.set_subnet()
    
def init_univ(args, model):
    i = 0
    gainProd = args.bfac
    for m in model.modules():
        if isinstance(m,(resnet_new.SupermaskLinear, nn.Linear)):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2/fan_in)
            if i==0:
                num = int(np.floor(fan_out/fan_in))
                ww = torch.zeros((fan_out, fan_in))
                bb = torch.zeros(fan_out)
                indRange = 0
                std = math.sqrt(2)
                gainProd = gainProd*std
                for j in range(fan_in):
                    nn.init.normal_(ww[indRange:(indRange+num),j], mean=0.0, std=std)
                    nn.init.normal_(bb[indRange:(indRange+num)], mean=0.0, std=gainProd)
                    indRange = num + indRange
                m.weight.data = ww
                m.bias.data = bb
            if i == 1:
                ww = torch.zeros((fan_out, fan_in))
                bb = torch.zeros(fan_out)
                std = math.sqrt(2)
                gainProd = gainProd*std
                for j in range(fan_in):
                    if j < fan_out:
                        nn.init.normal_(ww[j,j], mean=0.0, std=std)
                        nn.init.normal_(bb[j], mean=0.0, std=gainProd)
                m.weight.data = ww
                m.bias.data = bb
            if i == 2:
                ww = torch.zeros((fan_out, fan_in))
                bb = torch.zeros(fan_out)
                std = math.sqrt(2) #math.sqrt(2/num)
                indRange = 0
                gainProd = gainProd*std
                for j in range(fan_out):
                    nn.init.normal_(ww[j,indRange:(indRange+num)], mean=0.0, std=std)
                    nn.init.normal_(bb[j], mean=0.0, std=gainProd)
                    if j%num == 0 and j > 0:
                        if indRange+num < fan_in:
                            indRange = num + indRange
                m.weight.data = ww
                m.bias.data = bb
            if i > 2:
            #if i > 1:
                nn.init.normal_(m.weight, mean=0.0, std=std)
                m.weight.requires_grad = False
                gainProd = gainProd*std
                nn.init.normal_(m.bias, mean=0.0, std=gainProd)
            if args.zerobias:
                nn.init.zeros_(m.bias)
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            i = i+1
