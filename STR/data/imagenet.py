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

        trainset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform_test)
        self.val_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')
        # Data loading code
        

        


class ImageNet:
    def __init__(self, args):
        super(ImageNet, self).__init__()

        data_root = os.path.join(args.data, "imagenet")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )

class TinyImageNet:
    def __init__(self, args):
        super(TinyImageNet, self).__init__()

        data_root = os.path.join(args.data, "tiny_imagenet")
        
        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])

        train_dataset = H5DatasetOld(data_root + '/train.h5', transform=train_transforms)
        test_dataset = H5DatasetOld(data_root + '/val.h5', transform=test_transforms)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, transform=None):
        self.transform = transform
        self.dataFile = None
        self.h5_file = h5_file

    def __len__(self):
        datasetNames = list(self.dataFile.keys())
        return len(self.dataFile[datasetNames[0]])


    def __getitem__(self, idx):
        if self.dataFile is None:
            self.dataFile = h5py.File(self.h5_file, 'r')
        data = self.dataFile[list(self.dataFile.keys())[0]][idx]
        label = self.dataFile[list(self.dataFile.keys())[1]][idx]
        if self.transform:
            data = self.transform(data)
        return (data, label)

class H5DatasetOld(torch.utils.data.Dataset):
    def __init__(self, h5_file, transform=None):
        self.transform = transform
        self.dataFile = h5py.File(h5_file, 'r')
        # self.h5_file = h5_file

    def __len__(self):
        datasetNames = list(self.dataFile.keys())
        return len(self.dataFile[datasetNames[0]])


    def __getitem__(self, idx):
        # if self.dataFile is None:
        #     self.dataFile = h5py.File(self.h5_file, 'r')
        data = self.dataFile[list(self.dataFile.keys())[0]][idx]
        label = self.dataFile[list(self.dataFile.keys())[1]][idx]
        if self.transform:
            data = self.transform(data)
        return (data, label)