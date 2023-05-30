import os

import torch
from torchvision import datasets, transforms

import torch.multiprocessing
import h5py
import os
import numpy as np
torch.multiprocessing.set_sharing_strategy("file_system")


class CIFAR10:
    def __init__(self, args):
        super(CIFAR10, self).__init__()

        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True)

        testset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        self.val_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')
        # Data loading code
        


class CIFAR100:
    def __init__(self, args):
        super(CIFAR100, self).__init__()

        data_root = os.path.join(args.data, "cifar100")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        trainset = datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True)

        testset = datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        self.val_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False)

        

        