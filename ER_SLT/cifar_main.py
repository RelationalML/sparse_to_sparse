from __future__ import print_function
import argparse
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

from resnet_new import ResNet18
from vgg_new import vgg16, vgg16_bn
#from fc import tabular
import cifar_training

from initializers import *
from planting import *

import numpy as np

import sys
import pathlib
import time
args = None

def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, param in module.named_parameters():
        if "scores" in name:
            yield name, param

def get_weights(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, param in module.named_parameters():
        if "scores" in name:
            yield param

def load_dimension(dataset):
    if dataset == 'MNIST':
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == 'CIFAR10':
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == 'CIFAR100':
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == 'tiny-imagenet':
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == 'imagenet':
        input_shape, num_classes = (3, 224, 224), 1000
    #if dataset == 'class2':
    #    input_shape, num_classes = (1), 2
    #if dataset == 'tabular':
        
    return input_shape, num_classes

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def multistep_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        lr = args.lr * (0.1 ** (epoch // 30))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length


def main():
    global args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 edge-popup implementation')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train per level (default: 10)')
    parser.add_argument('--levels', type=int, default=10, metavar='N',
                        help='number of levels to anneal (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--warmup_length', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--scaling', action="store_true", default=False, help="scale parameters in every epoch")

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='how sparse is each layer')
    
    parser.add_argument('--sparsity_start', type=float, default=0.6,
                        help='initial sparsity of each layer')

    parser.add_argument('--anneal', type=bool, default=False,
                        help='whether sparsity should slowly be annealed towards target sparsity')
    
    parser.add_argument('--model', default='VGG16', choices=['Resnet18', 'VGG16', 'tabular'],
                        help='Type of model to use')
    
    parser.add_argument("--load_model", action="store_true", default=False, help="scale fan")
    
    parser.add_argument('--model_to_load', default='ticket_11_4.pt', 
                        help='Model to start pruning from.')
    
    parser.add_argument(
        "--initBias", default="kn-zero-bias", help="Bias initialization modifications",
        choices=["kn-nonzero-bias", "ortho-nonzero-bias", "kn-zero-bias", "ortho-bias-special", "ortho-zero-bias", "signed-const-with-bias", "signed-const-zero-bias"]
    )
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument("--dataset", default="CIFAR10", help="Data to train on.")
    
    parser.add_argument('--plant-model', action='store_true', default=False,
                        help='Plant based on dict specified by path-to-target.')
    
    parser.add_argument('--plant-path', type=str, default='./modelBest.pt',
                        help='Path to dict of model that should be planted.')

    parser.add_argument('--bfac', type=float, default=0.05, 
                        help='downscaling of bias initialization')
    
    parser.add_argument('--save_file', type=str, default='runs-cifar')

    parser.add_argument('--er-init-density', type = float, default=0.5)
    # parser.add_argument('--scaling', action="store_true", default=False, help="scale parameters in every epoch")


    #parser.add_argument('--depth', type=int, default=4, metavar='depth',
    #                    help='depth of fully-connected architecture')
    
    #parser.add_argument('--width', type=int, default=100, metavar='width',
    #                    help='width of fully-connected architecture')
    
    #parser.add_argument('--task', default='class', choices=['class', 'reg'], help='classification (class) or regression (reg) task')
    

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # device = torch.device("cpu")
    device = torch.device("cuda")
    args.workers = 1

    if not os.path.exists(args.save_file): os.makedirs(args.save_file)

    ## Dataset
    input_shape, num_classes = load_dimension(args.dataset)
    if (args.dataset == "CIFAR10"):
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.workers)

        testset = datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                shuffle=False, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                shuffle=False, num_workers=args.workers)
                
        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #if (args.dataset == "tabular"):
    #    
    #    #dataset = datagen.gen_data_helix_syn(is_train=train)
        
        
    ## Model
    if (args.model == "Resnet18"):
        model = ResNet18(args.sparsity, args.er_init_density, num_classes).to(device)
    elif (args.model == "VGG16"):
        model = vgg16_bn(args.sparsity, args.er_init_density, num_classes).to(device)
    #elif (args.model == "tabular"):
    #    model = tabular(args.sparsity, input_shape, num_classes, depth, width, task).to(device)


    ## Initialization
    if args.initBias == "kn-nonzero-bias":
        init_with_bias(args, model)
    if args.initBias == "ortho-nonzero-bias":
        init_with_bias_ortho(args, model)
    if args.initBias == "kn-zero-bias":
        init_zero_bias(args, model)
    if args.initBias == "ortho-bias-special":
        init_ortho_with_dep_bias(args, model)
    if args.initBias == "ortho-zero-bias":
        init_ortho_with_zero_bias(args, model)
    if args.initBias == "signed-const-with-bias":
        signed_const_with_bias(args, model)
    if args.initBias == "signed-const-zero-bias":
        signed_const_zero_bias(args, model)
    #load pretrained model from dict file
    if args.load_model:
        target_dict = torch.load(args.model_to_load)
        model.load_state_dict(target_dict)
    
    ## Planting
    if args.plant_model:
        plant_target_torch_fast(model, args.plant_path)
        print("Planting succeeded.")

    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        )

    
    #scheduler = CosineAnnealingLR(optimizer, T_max=10)
    scheduler = multistep_lr(optimizer, args)
    if (args.anneal):
        b = (args.sparsity/args.sparsity_start)**(1/args.levels)
        sparsity = args.sparsity_start
    for level in range(1, args.levels + 1):
        # anneal sparsity
        if (args.anneal):
            sparsity = sparsity*b #args.sparsity**(level / args.levels)

            # Annealing while making sure that the initial mask is ER, need to check how well this works
        
            if args.er_init_density <= sparsity:
                sparsity = args.er_init_density
            if sparsity < args.sparsity:
                sparsity = args.sparsity

            print ("Sparsity: " + str(sparsity))
            l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
            for layer in l:
                if isinstance(layer, (resnet_new.SupermaskLinear, resnet_new.SupermaskConv)):
                    layer.sparsity = sparsity
        acc_val_max = 0
        for epoch in range(1, args.epochs + 1):
            curr_train_acc = cifar_training.train(model, scheduler, device, train_loader, optimizer, loss, epoch, args.log_interval, args.scaling, args)
            acc_val = cifar_training.test(model, device, loss, val_loader)
            acc_test = cifar_training.test(model, device, loss, test_loader)

            #scheduler.step()
            if acc_val > acc_val_max:
                acc_val_max = acc_val
                best_acc = acc_test
                if args.save_model:
                    torch.save(model.state_dict(), "ticket_" + str(args.seed) + "_" + str(level) + ".pt")
            
            if epoch % 10 == 9:
                write_result_to_csv(
                    best_acc=best_acc,
                    curr_train_acc=curr_train_acc,
                    sparsity=args.sparsity,
                    sparsityCurr=sparsity,
                    curr_acc=acc_test,
                    dataset=args.dataset,
                    scaling=args.scaling,
                    init=args.initBias,
                    epochs=epoch,
                    anneal_epochs=args.levels,
                    seed=args.seed,
                    shuffle='Benchmark'
                )

    # if args.save_model:
    torch.save(model.state_dict(), args.save_file + "/ticket" + "_" + str(args.seed) + "_" + str(args.sparsity) + "_" + str(args.initBias) + "_" + str(args.scaling) + ".pt")
    
    '''

    # Here we add the ablation to shuffle the masks and see if learnt structure is important
    
    # first a sanity check
    for i in range(5):
        model.load_state_dict(torch.load(args.save_file + '/ticket' + '_' + str(args.seed) + '_' + str(args.sparsity) + '_' + str(args.initBias) + '_' + str(args.scaling) + '.pt', map_location = device))
        for name, param in model.named_parameters():
            if 'scores' in name:
                shape = param.shape
                num_zeros = (param == 0).sum()
                perm = torch.randperm(param.numel())
                param.data = param.data.view(-1)[perm].view(shape)
        acc_test = cifar_training.test(model, device, loss, test_loader)
        write_result_to_csv(
            best_acc=best_acc,
            curr_train_acc=curr_train_acc,
            sparsity=args.sparsity,
            sparsityCurr=sparsity,
            curr_acc=acc_test,
            dataset=args.dataset,
            scaling=args.scaling,
            init=args.initBias,
            epochs=args.epochs,
            anneal_epochs=args.levels,
            seed=args.seed,
            shuffle='Mask Shuffled'
        )
        torch.save(model.state_dict(), args.save_file + "/mask_shuffled_ticket" + "_" + str(args.seed) + "_" + str(args.sparsity) + "_" + str(args.initBias) + "_" + str(args.scaling) + "_run_" + str(i) + ".pt")


    for i in range(5):
        model.load_state_dict(torch.load(args.save_file + '/ticket' + '_' + str(args.seed) + '_' + str(args.sparsity) + '_' + str(args.initBias) + '_' + str(args.scaling) + '.pt', map_location = device))
        for (name, scores), param in zip(masks(model), get_weights(model)):
            shape = param.shape
            perm = torch.randperm(param.numel())
            param.data = param.data.view(-1)[perm].view(shape)
            scores.data = scores.data.view(-1)[perm].view(shape)
        acc_test = cifar_training.test(model, device, loss, test_loader)
        write_result_to_csv(
            best_acc=best_acc,
            curr_train_acc=curr_train_acc,
            sparsity=args.sparsity,
            sparsityCurr=sparsity,
            curr_acc=acc_test,
            dataset=args.dataset,
            scaling=args.scaling,
            init=args.initBias,
            epochs=args.epochs,
            anneal_epochs=args.levels,
            seed=args.seed,
            shuffle='Weight-Mask Shuffled'
        )
        torch.save(model.state_dict(), args.save_file + "/weight_mask_shuffled_ticket" + "_" + str(args.seed) + "_" + str(args.sparsity) + "_" + str(args.initBias) + "_" + str(args.scaling) + "_run_" + str(i) + ".pt")
'''




def write_result_to_csv(**kwargs):
    results = pathlib.Path(args.save_file) / "results.csv"

    if not results.exists():
        results.write_text(
            "Task, "
            "Initialization, "
            "Scaling, "
            "Epochs, "
            "Annealing epochs, "
            "Sparsity, "
            "Current sparsity, "
            "Current Train, "
            "Current Val, "
            "Best Val, "
            "Seed "
            "Shuffle \n"
        )

    now = time.time() #.strftime("%m-%d-%y_%H:%M:%S")
    #print(now)

    with open(results, "a+") as f:
        f.write(
            (
                "{dataset},"
                "{init},"
                "{scaling},"
                "{epochs},"
                "{anneal_epochs},"
                "{sparsity},"
                "{sparsityCurr},"
                "{curr_train_acc:.02f},"
                "{curr_acc:.02f},"
                "{best_acc:.02f},"
                "{seed}\n"
            ).format(**kwargs)
        )

if __name__ == '__main__':
    main()
