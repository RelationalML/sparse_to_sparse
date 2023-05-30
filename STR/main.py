'''
This code base is heavily built on the existing code base: https://github.com/RAIVNLab/STR
'''

import os
import pathlib
import random
import shutil
import time
import json
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import save_checkpoint, get_lr, LabelSmoothing
from utils.schedulers import get_policy, cosine_lr, assign_learning_rate, warmup_lr, constant_lr, multistep_lr
from utils.conv_type import STRConv, STRConvER, ConvER, ConvMask
from utils.conv_type import sparseFunction

from args import args
from trainer import train, validate, get_preds, hessian_trace

import data
from data import cifar10, imagenet

from models import resnet18


def main():
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    args.gpu = None

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    # model = get_model(args)
    if args.set == 'cifar10':
        if args.resnet_type == 'small-dense':
            model = resnet18.ResNetWidth18([3, 32, 32], num_classes=10, width = args.width)
        elif args.resnet_type == 'small-dense-inc':
            model = resnet18.ResNetWidth18Inc([3, 32, 32], num_classes=10, width = args.width)
        elif args.resnet_type == 'small-sub':
            model = resnet18.ResNetWidthSub18([3, 32, 32], num_classes=10, width = args.width)
        else:
            model = resnet18.ResNet18([3, 32, 32], num_classes=10)

    if args.set == 'cifar100':
        if args.resnet_type == 'small-dense':
            model = resnet18.ResNetWidth18([3, 32, 32], num_classes=100, width = args.width)
        elif args.resnet_type == 'small-dense-inc':
            model = resnet18.ResNetWidth18Inc([3, 32, 32], num_classes=100, width = args.width)
        elif args.resnet_type == 'small-sub':
            model = resnet18.ResNetWidthSub18([3, 32, 32], num_classes=100, width = args.width)
        elif args.resnet_type == 'res50':
            model = resnet18.ResNet50([3, 32, 32], num_classes=100)
        else:
            model = resnet18.ResNet18([3, 32, 32], num_classes=100)
    
    if args.set == 'tiny-imagenet':
        model = resnet.ResNet50(num_classes=200)


    if args.set == 'imagenet':
        model = resnet.ResNet50()

    if args.er_sparse_method == 'uniform':
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                m.set_er_mask(args.er_sparse_init)

        print(args.er_sparse_init)

    if args.er_sparse_method == 'ERK':
        sparsity_list = []
        num_params_list = []
        total_params = 0
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                sparsity_list.append(torch.tensor(m.weight.shape).sum() / m.weight.numel())
                num_params_list.append(m.weight.numel())
                total_params += m.weight.numel()
        
        num_params_kept = (torch.tensor(sparsity_list) * torch.tensor(num_params_list)).sum()
        num_params_to_keep = total_params * args.er_sparse_init
        C = num_params_to_keep / num_params_kept
        sparsity_list = [torch.clamp(C*s, 0, 1) for s in sparsity_list]
        print(sparsity_list)
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                m.set_er_mask(sparsity_list[l])
                l += 1
        print(sparsity_list)
        

    if args.er_sparse_method == 'balanced':
        total_params = 0
        l = 0
        sparsity_list = []
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                total_params += m.weight.numel()
                l += 1

        X = args.er_sparse_init * total_params / l

        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                if X / m.weight.numel() < 1.0:
                    sparsity_list.append(X / m.weight.numel())
                else: 
                    sparsity_list.append(1)

        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                m.set_er_mask(sparsity_list[l])
                l += 1
        print(sparsity_list)

    if args.er_sparse_method == 'pyramidal':

        num_params = 0
        layerwise_params = []
        for name, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                num_params += m.weight.numel()
                layerwise_params.append(m.weight.numel())
        layerwise_params = layerwise_params[::-1]
        layerwise_params.append(-num_params*args.er_sparse_init)
        roots = np.roots(np.array(layerwise_params))
        print('Roots of np solver', roots)
        for r in roots:
            if r < 1 and r > 0 and np.imag(r) == 0:
                print('root identified: ', r)
                layerwise_sparsities = np.arange(1, len(layerwise_params) + 1)
                sparsity_list = np.real(r) ** layerwise_sparsities
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                m.set_er_mask(sparsity_list[l])
                l += 1
        
        print(layerwise_sparsities)
    
    if args.er_sparse_method == 'str-induced-er':
        with open('runs/layerwise_sparsity/' + args.er_sparsity_file) as f:
           data = json.load(f)
        sparsity_list = list(data.values())
        alpha = sparsity_list[-1] / args.er_sparse_init
        sparsity_list = [s / alpha for s in sparsity_list[:-1]]
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (STRConvER, ConvER, ConvMask)):
                m.set_er_mask(sparsity_list[l])
                l += 1
        print(sparsity_list)
        


    

    model = set_gpu(args, model)
    print('The model definition is:')
    print(model)
    total_num_narrow = 0
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total_num_narrow += m.weight.numel()
    print('num params', total_num_narrow)
    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)

    

    optimizer = get_optimizer(args, model)
    # data = get_dataset(args)
    if args.set == 'cifar10':  
        data = cifar10.CIFAR10(args)
    if args.set == 'cifar100':
        data = cifar10.CIFAR100(args)
    if args.set == 'tiny-imagenet':
        data = cifar10.TinyImagenet(args)
    if args.set == 'imagenet':
        data = imagenet.ImageNet(args)
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    
    # Evaulation of a model
    if args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )
        return

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    



    # random prune iterative
    def prune_random(model, density):
        
        total_params = 0
        l = 0
        sparsity_list = []
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                total_params += m.weight.numel()
                l += 1

        X = density * total_params / l

        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                if X / m.weight.numel() < 1.0:
                    sparsity_list.append(X / m.weight.numel())
                else: 
                    sparsity_list.append(1)
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                curr_nz = (m.mask == 1).sum()
                target_nz = int(sparsity_list[cnt] * m.mask.numel())
                print(curr_nz, target_nz, m.mask.numel())
                diff = target_nz / (curr_nz - target_nz) 
                if diff > 0 and diff < 1:
                    m.mask = torch.where(m.mask == 0, m.mask, torch.empty(m.mask.shape).bernoulli_(diff))
                cnt+=1
        return model
    

    def prune_mag(model, density):
        score_list = {}
        for n, m in model.named_modules():
            # torch.cat([torch.flatten(v) for v in self.scores.values()])
            if isinstance(m, (ConvMask)):
                score_list[n] = (m.mask.to(m.weight.device) * m.weight).detach().abs_()

        global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
        k = int((1 - density) * global_scores.numel())
        threshold, _ = torch.kthvalue(global_scores, k)

        if not k < 1:
            total_num = 0
            total_den = 0
            for n, m in model.named_modules():
                if isinstance(m, (ConvMask)):
                    score = score_list[n].to(m.weight.device)
                    zero = torch.tensor([0.]).to(m.weight.device)
                    one = torch.tensor([1.]).to(m.weight.device)
                    m.mask = torch.where(score <= threshold, zero, one)
                    total_num += (m.mask == 1).sum()
                    total_den += m.mask.numel()

        print('Overall model density after magnitude pruning at current iteration = ', total_num / total_den)
        return model

    def prune_random_balanced(model, density):

        total_params = 0
        l = 0
        sparsity_list = []
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                total_params += m.weight.numel()
                l += 1

        X = density * total_params / l
        score_list = {}

        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score_list[n] = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()

                if X / m.weight.numel() < 1.0:
                    sparsity_list.append(X / m.weight.numel())
                else: 
                    sparsity_list.append(1)

        total_num = 0
        total_den = 0
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                global_scores = torch.flatten(score_list[n])
                k = int((1 - sparsity_list[cnt]) * global_scores.numel())
                if k == 0:
                    threshold = 0
                else: 
                    threshold, _ = torch.kthvalue(global_scores, k)
                print('Layer', n, ' params ', k, global_scores.numel())

                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
                cnt += 1

        print('Overall model density after random global (balanced) pruning at current iteration = ', total_num / total_den)
        return model


    def prune_random_uniform(model, density):

        total_num = 0
        total_den = 0
        
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()
                global_scores = torch.flatten(score)
                k = int((1 - density) * global_scores.numel())
                if k == 0:
                    threshold = 0
                else: 
                    threshold, _ = torch.kthvalue(global_scores, k)
                print('Layer', n, ' params ', k, global_scores.numel())

                score = score.to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
                

        print('Overall model density after random global (uniform) pruning at current iteration = ', total_num / total_den)
        return model


    def prune_random_erk(model, density):

        sparsity_list = []
        num_params_list = []
        total_params = 0
        score_list = {}

        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score_list[n] = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()

                sparsity_list.append(torch.tensor(m.weight.shape).sum() / m.weight.numel())
                num_params_list.append(m.weight.numel())
                total_params += m.weight.numel()
        
        num_params_kept = (torch.tensor(sparsity_list) * torch.tensor(num_params_list)).sum()
        num_params_to_keep = total_params * density
        C = num_params_to_keep / num_params_kept
        print('Factor: ', C)
        sparsity_list = [torch.clamp(C*s, 0, 1) for s in sparsity_list]

        total_num = 0
        total_den = 0
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                global_scores = torch.flatten(score_list[n])
                k = int((1 - sparsity_list[cnt]) * global_scores.numel())
                if k == 0:
                    threshold = 0
                else: 
                    threshold, _ = torch.kthvalue(global_scores, k)
                print('Layer', n, ' params ', k, global_scores.numel())

                score = score_list[n].to(m.weight.device)
                zero = torch.tensor([0.]).to(m.weight.device)
                one = torch.tensor([1.]).to(m.weight.device)
                m.mask = torch.where(score <= threshold, zero, one)
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
                cnt += 1

        print('Overall model density after random global (ERK) pruning at current iteration = ', total_num / total_den)
        return model


    def prune_random_global(model, density):

        score_list = {}

        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score_list[n] = (m.mask.to(m.weight.device) * torch.randn_like(m.weight).to(m.weight.device)).detach().abs_()

        global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
        k = int((1 - density) * global_scores.numel())
        threshold, _ = torch.kthvalue(global_scores, k)

        total_num = 0
        total_den = 0
        cnt = 0
        if not k < 1:
            for n, m in model.named_modules():
                if isinstance(m, (ConvMask)):
                    
                    score = score_list[n].to(m.weight.device)
                    zero = torch.tensor([0.]).to(m.weight.device)
                    one = torch.tensor([1.]).to(m.weight.device)
                    m.mask = torch.where(score <= threshold, zero, one)
                    total_num += (m.mask == 1).sum()
                    total_den += m.mask.numel()
                    cnt += 1

        print('Overall model density after random global pruning at current iteration = ', total_num / total_den)
        return model

    def prune_snip(model, trainloader, loss, density):

        for i, (images, target) in enumerate(trainloader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True).long()
            model.zero_grad()
            output = model(images)
            loss(output, target).backward()
            break
        
        score_list = {}
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score_list[n] = (m.weight.grad * m.weight * m.mask.to(m.weight.device)).detach().abs_()
        
        global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
        k = int((1 - density) * global_scores.numel())
        threshold, _ = torch.kthvalue(global_scores, k)

        if not k < 1:
            total_num = 0
            total_den = 0
            for n, m in model.named_modules():
                if isinstance(m, (ConvMask)):
                    score = score_list[n].to(m.weight.device)
                    zero = torch.tensor([0.]).to(m.weight.device)
                    one = torch.tensor([1.]).to(m.weight.device)
                    m.mask = torch.where(score <= threshold, zero, one)
                    total_num += (m.mask == 1).sum()
                    total_den += m.mask.numel()

        print('Overall model density after snip pruning at current iteration = ', total_num / total_den)
        return model


    def prune_synflow(model, trainloader, density):

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        signs = linearize(model)

        (data, _) = next(iter(trainloader))
        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(args.gpu)#, dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()
        
        score_list = {}
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                score_list[n] = (m.mask.to(m.weight.device) * m.weight.grad * m.weight).detach().abs_()
        
        model.zero_grad()

        nonlinearize(model, signs)

        global_scores = torch.cat([torch.flatten(v) for v in score_list.values()])
        k = int((1 - density) * global_scores.numel())
        threshold, _ = torch.kthvalue(global_scores, k)

        if not k < 1:
            total_num = 0
            total_den = 0
            for n, m in model.named_modules():
                if isinstance(m, (ConvMask)):
                    score = score_list[n].to(m.weight.device)
                    zero = torch.tensor([0.]).to(m.weight.device)
                    one = torch.tensor([1.]).to(m.weight.device)
                    m.mask = torch.where(score <= threshold, zero, one)
                    total_num += (m.mask == 1).sum()
                    total_den += m.mask.numel()

        print('Overall model density after synflow pruning at current iteration = ', total_num / total_den)
        return model


    def dead_neurons(model):
        dead_list = []
        dead_mask_list_in = []
        dead_mask_list_out = []
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                idxs = (m.mask.sum(dim=(1,2,3)) == 0).nonzero(as_tuple=True)
                dead_mask_in = torch.where(m.mask.sum(dim=(1,2,3)) == 0, 1, 0)
                dead_mask_list_in.append(dead_mask_in)

                dead_mask_out = torch.where(m.mask.sum(dim=(0,2,3)) == 0, 1, 0)
                dead_mask_list_out.append(dead_mask_out)

                num_dead = idxs[0].numel()
                dead_ratio = num_dead/m.mask.shape[0]
                dead_list.append((num_dead, dead_ratio))
            
        return dead_list, dead_mask_list_in, dead_mask_list_out

    def prune_dead(model):
        dead_idx_in = []
        dead_idx_out = []
        num_in = 0
        num_out = 0
        print('if there are any dead neurons in the network, all the incoming and outgoing weights to it are zeroed out')
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                in_idxs = (m.mask.sum(dim=(1,2,3)) == 0).nonzero(as_tuple=True)
                dead_idx_in.append(in_idxs)
                num_in += in_idxs[0].shape[0]
                out_idxs = (m.mask.sum(dim=(0,2,3)) == 0).nonzero(as_tuple=True)
                dead_idx_out.append(out_idxs)
                num_out += out_idxs[0].shape[0]

        print('Num in and num out: ', num_in, num_out)
        # Removing dead incoming neurons
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                if l == 0:
                    idx = dead_idx_in[l]
                    m.mask[idx, :, :, :] = 0
                elif l == len(dead_idx_in)-1:
                    pass
                else:
                    idx = dead_idx_in[l]
                    m.mask[idx, :, :, :] = 0
                    idx_out = dead_idx_in[l-1]
                    m.mask[:, idx, :, :] = 0

                l += 1

        # Removing dead outgoing neurons
        l = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                if l == 0:
                    idx = dead_idx_out[l]
                    m.mask[:, idx, :, :] = 0
                elif l == len(dead_idx_out)-1:
                    pass
                else:
                    idx = dead_idx_out[l]
                    m.mask[:,idx, :, :] = 0
                    idx_out = dead_idx_out[l-1]
                    m.mask[idx, :, :, :] = 0

                l += 1

        return model

    


    def prune_mask_ref(model, level, name):
        # Loading a mask at every level
        # ref_mask = 'runs/mask_cifar-imp-rewind-save-every-0.1-seed-4_'
        
            
        ref_mask = base_dir + 'runs/mask_' + str(name) + '_'
        mask_list = torch.load(ref_mask + str(level) + '.pt')

        total_num = 0
        total_den = 0
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                
                m.mask = mask_list[cnt]
                total_num += (m.mask == 1).sum()
                total_den += m.mask.numel()
                cnt += 1
    
        print('Overall model density after loading IMP mask = ', total_num / total_den)
        return model

    # Iterative Prune Train before final train
    
    warmup_scheduler = warmup_lr(optimizer, args)
    val_acc_total = []

    ############## 
    base_dir = ''
    
        
    # warmup training
    if not args.warmup:
        args.warmup_epochs = 0
    curr_epoch = 0
    if args.warmup:
        print('Warm Up training for the model')
        for epoch in range(args.warmup_epochs):
            warmup_scheduler(epoch)
            lr = get_lr(optimizer)
            print('The curent learning rate is: ', lr)

            start_train = time.time()
            train_acc1, train_acc5 = train(
                data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
            )
            train_time.update((time.time() - start_train) / 60)
            # evaluate on validation set
            start_validation = time.time()
            acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
            validation_time.update((time.time() - start_validation) / 60)
            
            val_acc_total.append(acc1)
            curr_epoch += 1

    # save the model and the optimizer
    torch.save(model.state_dict(), "{}runs/model_{}_init.pt".format(base_dir, args.name))
    torch.save(optimizer.state_dict(),"{}runs/optimizer_{}.pt".format(base_dir, args.name))
    torch.save(val_acc_total, 'runs/val_acc_'+ args.name + '.pt')

    hessian_list = []

    ####################
    # Load a model and mask from a given checkpoint
    if args.load_before_prune:
        print('Loading model from: ', args.load_model_name, args.load_mask_name)
        model = resume_from_checkpoint(model, args.load_model_name, args.load_mask_name)
    
    # Loads only the model, mask will be all ones
    if args.load_only_model:
        print('Loading ONLY model from: ', args.load_model_name)
        base_dir = ''

        model_name = base_dir + 'runs/' + args.load_model_name
        model.load_state_dict(torch.load(model_name))
        # Save the model again as init so that it can be loaded for reset weights
        print('over writing model weights init with loaded model')
        torch.save(model.state_dict(), "{}runs/model_{}_init.pt".format(base_dir, args.name))
        ####################

    # Loads only the model, mask will be all ones and also loads the sign for the initialized model
    if args.load_model_mask_and_sign:
        print('Loading ONLY model and mask from: ', args.load_model_name, args.load_mask_name)
        base_dir = ''
        
        model = resume_from_checkpoint(model, args.load_model_name, args.load_mask_name)

        print('Changing the sign of the reset weight according to the expt: ', args.load_expt_sign)
        sign_list = torch.load("{}runs/sign_list_{}.pt".format(base_dir, args.load_expt_sign))
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                m.weight.data = m.weight.data.abs_() * sign_list[cnt].to(m.weight.device)
                cnt += 1

        # Save the model again as init so that it can be loaded for reset weights
        print('over writing model weights init with loaded model')
        torch.save(model.state_dict(), "{}runs/model_{}_sign_changed_init.pt".format(base_dir, args.name))

    ######
    if args.load_shuffled_model_mask_and_sign:
        print('Loading model and mask from, and shuffling the model weights: ', args.load_model_name, args.load_mask_name)
        base_dir = ''
        
        model = resume_from_checkpoint(model, args.load_model_name, args.load_mask_name)

        print('Changing the sign of the reset weight according to the expt: ', args.load_expt_sign)
        sign_list = torch.load("{}runs/sign_list_{}.pt".format(base_dir, args.load_expt_sign))
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, (ConvMask)):
                curr_shape = m.weight.shape
                idx = torch.where(m.mask == 1)
                buf = m.weight.data.clone()
                shuffle = buf[idx]
                perm = torch.randperm(shuffle.numel())
                shuffle = shuffle[perm]
                buf[idx] = shuffle
                
                m.weight.data = buf.abs_() * sign_list[cnt].to(m.weight.device)
                cnt += 1

        # Save the model again as init so that it can be loaded for reset weights
        print('over writing model weights init with loaded model')
        torch.save(model.state_dict(), "{}runs/model_{}_model_shuffled_sign_changed_init.pt".format(base_dir, args.name))


    if args.hessian:
        trace = hessian_trace(data.train_loader, model, criterion, optimizer, args)
        hessian_list.append(trace)
    
    level = 0
    if args.conv_type == 'ConvMask':
        print('Threshold list: ', args.threshold_list)
        for threshold in args.threshold_list:
            assign_learning_rate(optimizer, args.lr)
            if args.prune_scheduler == 'cosine':
                prune_scheduler = cosine_lr(optimizer, args.final_prune_epoch, args)
            if args.prune_scheduler == 'constant':
                prune_scheduler = constant_lr(optimizer, args)
                assign_learning_rate(optimizer, args.constant_prune_lr)
                print('Learning rate during pruning is constant at: ', args.constant_prune_lr)
            if args.prune_scheduler == 'step':
                prune_scheduler = multistep_lr(optimizer, args)
            
            for epoch in range(args.final_prune_epoch):
                prune_scheduler(epoch)
                lr = get_lr(optimizer)
                print('The curent learning rate is: ', lr)

                start_train = time.time()
                train_acc1, train_acc5 = train(
                    data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
                )
                train_time.update((time.time() - start_train) / 60)

                # evaluate on validation set
                start_validation = time.time()
                acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
                validation_time.update((time.time() - start_validation) / 60)
                val_acc_total.append(acc1)
                curr_epoch += 1
                if args.hessian:
                    trace = hessian_trace(data.train_loader, model, criterion, optimizer, args)
                    hessian_list.append(trace)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)
             
            
            if args.pruner == 'random-balanced':
                print('Pruning randomly with balanced at density: ', threshold)
                model = prune_random_balanced(model, threshold)
            
            if args.pruner == 'random-global':
                print('Pruning globally randomly at density: ', threshold)
                model = prune_random_global(model, threshold) 
            
            if args.pruner == 'random-erk':
                print('Pruning with random (ERK) with density: ', threshold)
                model = prune_random_erk(model, threshold)

            if args.pruner == 'random-uniform':
                print('Pruning with random (Uniform) with density: ', threshold)
                model = prune_random_uniform(model, threshold) 

            if args.pruner == 'mag':
                print('Pruning by magnitude at density: ', threshold)
                model = prune_mag(model, threshold)    

            if args.pruner == 'snip':
                print('Pruning by SNIP at density: ', threshold)
                model = prune_snip(model, data.train_loader, criterion, threshold)     
            
            if args.pruner == 'synflow':
                print('Pruning by SynFlow at density: ', threshold)
                model = prune_synflow(model, data.train_loader, threshold)     
            
            if args.pruner == 'mask-ref':
                print('Pruning by loading an equivalent mask from a different experiment')
                model = prune_mask_ref(model, level, args.target_expt_name)

            

            mask_list = []
            for name, m in model.named_modules():
                if isinstance(m, (ConvMask)):
                    mask_list.append(m.mask)
            torch.save(mask_list, 'runs/mask_{}_{}.pt'.format(args.name, level))
            torch.save(model.state_dict(),"runs/model_{}_{}.pt".format(args.name, level))
            level += 1

            torch.save(val_acc_total, 'runs/val_acc_'+ args.name + '.pt')

            if args.reset_weights:
                original_dict = torch.load("{}runs/model_{}_init.pt".format(base_dir, args.name))
                original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}runs/optimizer_{}.pt".format(base_dir, args.name)))
                print('Weights of the model reset to initialization weights')
                
                if args.reset_weight_sign:
                    print('Changing the sign of the reset weight according to the expt: ', args.load_expt_sign)
                    sign_list = torch.load("{}runs/sign_list_{}_{}.pt".format(base_dir, args.load_expt_sign, level))
                    cnt = 0
                    for n, m in model.named_modules():
                        if isinstance(m, (ConvMask)):
                            m.weight.data = m.weight.data.abs_() * sign_list[cnt].to(m.weight.device)
                            cnt += 1
            
            if args.reset_only_bn:
                print('resetting BN parameters and the optimizer')
                original_dict = torch.load("{}runs/model_{}_init.pt".format(base_dir, args.name))
                # original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                original_weights = {k:v for (k,v) in original_dict.items() if ('bn' in k) or ('downsample.1' in k)}
                print('resets the BN parameters along with the running mean and variance to the init values')
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}runs/optimizer_{}.pt".format(base_dir, args.name)))
               
            if args.reset_weight_amplitude:

                sign_list = []
                for n, m in model.named_modules():
                    if isinstance(m, (ConvMask)):
                        sign_list.append(m.weight.sign())

                original_dict = torch.load("{}runs/model_{}_init.pt".format(base_dir, args.name))
                original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}runs/optimizer_{}.pt".format(base_dir, args.name)))
                print('Resetting only the amplitude during IMP, while keeping the signs!!!')

                cnt = 0
                for n, m in model.named_modules():
                    if isinstance(m, (ConvMask)):
                        m.weight.data = m.weight.data.abs_() * sign_list[cnt].to(m.weight.device)
                        cnt += 1

            if args.reset_only_weight_sign:
                # Get the signs from the initial model.
                original_dict = torch.load("{}runs/model_{}_init.pt".format(base_dir, args.name))
                sign_list = []
                for n in original_dict.keys():
                    if 'weight' in n and(('conv' in n) or ('linear' in n) or ('downsample.0.' in n)):
                        sign_list.append(original_dict[n].sign())

                optimizer.load_state_dict(torch.load("{}runs/optimizer_{}.pt".format(base_dir, args.name)))
                print('resetting only the signs of the weights to the signs of IMP init and continuing LRR')
                cnt = 0
                for n, m in model.named_modules():
                    if isinstance(m, (ConvMask)):
                        m.weight.data = m.weight.data.abs_() * sign_list[cnt].to(m.weight.device)
                        cnt += 1

    # save the mask of the sparse structure
    mask_list = []
    total_num = 0
    total_den = 0
    for name, m in model.named_modules():
        if isinstance(m, (ConvMask)):
            mask_list.append(m.mask)
            total_num += m.mask.sum()
            total_den += m.mask.numel()
    print('Density before full training is: ', total_num / total_den)
    torch.save(mask_list, 'runs/mask_{}.pt'.format(args.name))


    # Start training
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0
    
    # Final training for the pruned network follows a cosine LR schedule
    assign_learning_rate(optimizer, args.lr)
    lr_scheduler = cosine_lr(optimizer, args.epochs, args)

    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler(epoch)
        cur_lr = get_lr(optimizer)
        print('The curent learning rate is: ', cur_lr)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)
        val_acc_total.append(acc1)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        curr_epoch += 1
        
        if args.hessian:
            trace = hessian_trace(data.train_loader, model, criterion, optimizer, args)
            hessian_list.append(trace)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            
        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )

        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()
        torch.save(val_acc_total, 'runs/val_acc_'+ args.name + '.pt')    

        # Storing sparsity and threshold statistics for STRConv models
        if args.conv_type == 'STRConv' or args.conv_type == 'STRConvER' or args.conv_type == 'ConvER' or args.conv_type == 'ConvMask':
            count = 0
            sum_sparse = 0.0
            for n, m in model.named_modules():
                if isinstance(m, (STRConv, STRConvER, ConvER, ConvMask)):
                    sparsity, total_params, thresh = m.getSparsity()
                    writer.add_scalar("sparsity/{}".format(n), sparsity, epoch)
                    writer.add_scalar("thresh/{}".format(n), thresh, epoch)
                    sum_sparse += int(((100 - sparsity) / 100) * total_params)
                    count += total_params
            total_sparsity = 100 - (100 * sum_sparse / count)
            writer.add_scalar("sparsity/total", total_sparsity, epoch)
        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

    torch.save(hessian_list, 'runs/hessian_{}.pt'.format(args.name))
    torch.save(model.state_dict(),"runs/model_{}_trained.pt".format(args.name))

    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        prune_rate=args.prune_rate,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=args.config,
        name=args.name,
        sparsity=total_sparsity,
    )

    

    if args.conv_type == "STRConv" or args.conv_type == 'STRConvER' or args.conv_type == 'ConvER' or args.conv_type == 'ConvMask':
        json_data = {}
        json_thres = {}
        for n, m in model.named_modules():
            if isinstance(m, (STRConv, STRConvER, ConvER, ConvMask)):
                sparsity = m.getSparsity()
                json_data[n] = sparsity[0]
                sum_sparse += int(((100 - sparsity[0]) / 100) * sparsity[1])
                count += sparsity[1]
                json_thres[n] = sparsity[2]
        json_data["total"] = 100 - (100 * sum_sparse / count)
        if not os.path.exists("runs/layerwise_sparsity"):
            os.mkdir("runs/layerwise_sparsity")
        if not os.path.exists("runs/layerwise_threshold"):
            os.mkdir("runs/layerwise_threshold")
        with open("runs/layerwise_sparsity/{}.json".format(args.name), "w") as f:
            json.dump(json_data, f)
        with open("runs/layerwise_threshold/{}.json".format(args.name), "w") as f:
            json.dump(json_thres, f)


def set_gpu(args, model):
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def resume_from_checkpoint(model, model_name, mask_name):
    base_dir = ''
    

    model_name = base_dir + 'runs/' + model_name
    mask_name = base_dir + 'runs/' + mask_name
    model.load_state_dict(torch.load(model_name))
    mask_list = torch.load(mask_name)
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, ConvMask):
            m.mask = mask_list[cnt].to(m.weight.device)
            cnt += 1

    return model




def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]

        model_state_dict = model.state_dict()

        if not args.ignore_pretrained_weights:

            pretrained_final = {
                k: v
                for k, v in pretrained.items()
                if (k in model_state_dict and v.size() == model_state_dict[k].size())
            }

            if args.conv_type != "STRConv":
                for k, v in pretrained.items():
                    if 'sparseThreshold' in k:
                        wkey = k.split('sparse')[0] + 'weight'
                        weight = pretrained[wkey]
                        pretrained_final[wkey] = sparseFunction(weight, v)

            model_state_dict.update(pretrained_final)
            model.load_state_dict(model_state_dict)

        # Using the budgets of STR models for other models like DNW and GMP
        if args.use_budget:
            budget = {}
            for k, v in pretrained.items():
                if 'sparseThreshold' in k:
                    wkey = k.split('sparse')[0] + 'weight'
                    weight = pretrained[wkey]
                    sparse_weight = sparseFunction(weight, v)
                    budget[wkey] = (sparse_weight.abs() > 0).float().mean().item()

            for n, m in model.named_modules():
                if hasattr(m, 'set_prune_rate'):
                    pr = 1 - budget[n + '.weight']
                    m.set_prune_rate(pr)
                    print('set prune rate', n, pr)


    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):

    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    print(f"=> Num model params {sum(p.numel() for p in model.parameters())}")

    

        

    # applying sparsity to the network
    if args.conv_type != "DenseConv":

        print(f"==> Setting prune rate of network to {args.prune_rate}")

        def _sparsity(m):
            if hasattr(m, "set_prune_rate"):
                m.set_prune_rate(args.prune_rate)

        model.apply(_sparsity)

    # freezing the weights if we are only doing mask training
    if args.freeze_weights:
        print(f"=> Freezing model weights")

        def _freeze(m):
            if hasattr(m, "mask"):
                m.weight.requires_grad = False
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = False

        model.apply(_freeze)

    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            pass #print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            pass #print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        sparse_thresh = [v for n, v in parameters if ("sparseThreshold" in n) and v.requires_grad]
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        # rest_params = [v for n, v in parameters if ("bn" not in n) and ('sparseThreshold' not in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and ("sparseThreshold" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {
                    "params": sparse_thresh,
                    "weight_decay": args.st_decay if args.st_decay is not None else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    filename = args.result_dir + '.csv'
    base_dir = ''
    

    results = pathlib.Path(base_dir, "runs", filename)
    
    if not results.exists():
        with open(results, 'w', newline=''):

            results.write_text(
                "Date Finished, "
                "Base Config, "
                "Name, "
                "Prune Rate, "
                "Current Val Top 1, "
                "Current Val Top 5, "
                "Best Val Top 1, "
                "Best Val Top 5, "
                "Best Train Top 1, "
                "Best Train Top 5,"
                "Sparsity\n"
            )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f},"
                "{sparsity:.04f}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()
