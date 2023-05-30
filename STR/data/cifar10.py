import os

import torch
from torchvision import datasets, transforms

import torch.multiprocessing
from torch.utils.data import Dataset
import math
# import h5py
import os
import numpy as np
torch.multiprocessing.set_sharing_strategy("file_system")

class MNIST:
    def __init__(self, args):
        super(MNIST, self).__init__()

        data_root = os.path.join(args.data, "mnist")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        
        transform_train = transforms.Compose([
                            transforms.ToTensor(), # first, convert image to PyTorch tensor
                            transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                            ])

        transform_test = transforms.Compose([
                            transforms.ToTensor(), # first, convert image to PyTorch tensor
                            transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                            ])

        trainset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True)

        self.train_loader_pred = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=False)

        testset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_test)
        self.val_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False)

class TinyImagenet:
    def __init__(self, args):
        super(TinyImagenet, self).__init__()

        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
        ])

        trainset = datasets.ImageFolder(
            './data/tiny-imagenet-200/train', transform=transform_train)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=12)
        

        testset = datasets.ImageFolder(
            root='./data/tiny-imagenet-200/val', transform=transform_test)

        self.train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False)
        
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

        self.train_loader_pred = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=False)

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

        



class TicketDataset(Dataset):

    def __init__(self, data, d_in, train_proportion, val_proportion, is_training, is_validation):

        if is_training:
            ran = np.arange(0,math.ceil(train_proportion*data.shape[0]))
        if is_validation:
            ran = np.arange(math.ceil(train_proportion*data.shape[0]),math.ceil((train_proportion+val_proportion)*data.shape[0]))
        else:
            ran = np.arange(math.ceil((train_proportion+val_proportion)*data.shape[0]),data.shape[0])
        m = data.shape[1] - d_in
        self.data_feats = torch.Tensor(data[ran[:,np.newaxis],np.arange(d_in)[np.newaxis,:]])
        self.data_resp = torch.Tensor(data[ran[:,np.newaxis],np.arange(d_in,d_in + m)[np.newaxis,:]])
        self.d_in = d_in
        self.d_out = m
        #print(self.data_resp.shape)

    def __len__(self):
        return self.data_feats.shape[0]


    def __getitem__(self, index):
        return self.data_feats[index,:], self.data_resp[index].squeeze()


def load_file(file_name='heart_dataset/heartDL.txt', d_in=6, seed=42, is_train = True, is_val = False, split = .8, splitVal = .1):
    #d_in: number of features, remaining (last) columns correspond to target
    data = np.loadtxt(file_name, delimiter=',', skiprows=0)  #draw_data_helix(n, 1, noise)
    n = data.shape[0]
    ## shuffle with fixed seed for same effect across datasets of synflow
    perm = np.random.RandomState(seed=seed).permutation(n)
    data = data[perm,]
    
    dataset = TicketDataset(data, d_in, split, splitVal, is_train, is_val)
    
    return dataset

class HeartData:
    def __init__(self, args):
        super(HeartData, self).__init__()


        trainset = load_file(file_name='heart_dataset/heartDLcont.txt', d_in=6, seed=42, is_train = True, is_val = False, split = .8, splitVal = .1)
        self.train_loader = torch.utils.data.DataLoader(dataset=trainset, 
                                             batch_size=args.batch_size, 
                                             shuffle=True)
        
        testset = load_file(file_name='heart_dataset/heartDLcont.txt', d_in=6, seed=42, is_train = False, is_val = True, split = .8, splitVal = .1)
        self.val_loader = torch.utils.data.DataLoader(dataset=testset, 
                                             batch_size=10, 
                                             shuffle=True)