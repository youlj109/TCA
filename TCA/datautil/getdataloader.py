# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader
import torchvision
import torch
import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader
from PIL import Image
import os
from torch.utils.data import ConcatDataset
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CustomCIFAR100Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.labels = [int(label) for label in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, label))]
        self.image_paths = []
        self.image_labels = []
        
        for label in self.labels:
            label_folder = os.path.join(root_dir, str(label))
            for img_name in os.listdir(label_folder):
                self.image_paths.append(os.path.join(label_folder, img_name))
                self.image_labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.image_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label
                
def get_img_dataloader(args):
    rate = 0.2

    if args.dataset in ['CIFAR-10', 'CIFAR-100', 'ImageNet']:
        print('Corrupt Dataset\t')
        if args.dataset=='CIFAR-10':
            transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

            transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

            trainset = torchvision.datasets.CIFAR10(root='/root/autodl-tmp/TTA_data', train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            testset = torchvision.datasets.CIFAR10(root='/root/autodl-tmp/TTA_data', train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            return trainloader, testloader
        elif args.dataset == 'CIFAR-100':
            transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

            transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

            trainset = CustomCIFAR100Dataset(root_dir=os.path.join(args.data_dir, 'train'), transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            testset = CustomCIFAR100Dataset(root_dir=os.path.join(args.data_dir, 'test'), transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            return trainloader, testloader
    
    else :
        trdatalist, tedatalist = [], []
        names = args.img_dataset[args.dataset]
        args.domain_num = len(names)
        for i in range(len(names)): 
            if i in args.test_envs:
                tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                            names[i], i, transform=imgutil.image_test(args.dataset), test_envs=args.test_envs))
            else:
                tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
                                        names[i], i, transform=imgutil.image_train(args.dataset), test_envs=args.test_envs).labels
                l = len(tmpdatay)
                if args.split_style == 'strat':
                    lslist = np.arange(l)
                    stsplit = ms.StratifiedShuffleSplit(
                        2, test_size=rate, train_size=1-rate, random_state=args.seed)
                    stsplit.get_n_splits(lslist, tmpdatay)
                    indextr, indexte = next(stsplit.split(lslist, tmpdatay))
                else:
                    indexall = np.arange(l)
                    np.random.seed(args.seed)
                    np.random.shuffle(indexall)
                    ted = int(l*rate)
                    indextr, indexte = indexall[:-ted], indexall[-ted:]

                trdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                            names[i], i, transform=imgutil.image_train(args.dataset), indices=indextr, test_envs=args.test_envs))
                tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                            names[i], i, transform=imgutil.image_test(args.dataset), indices=indexte, test_envs=args.test_envs))

        train_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=args.batch_size,
            num_workers=args.N_WORKERS)
            for env in trdatalist]

        eval_loaders = [DataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False)
            for env in trdatalist+tedatalist]

        return train_loaders, eval_loaders

def get_fisher_dataset(args):
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)

    train_datasets = []
    for i in range(len(names)):
        if i not in args.test_envs:
            train_dir = os.path.join(args.data_dir,names[i])
            dataset = datasets.ImageFolder(train_dir, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
            train_datasets.append(dataset)

    fisher_dataset = ConcatDataset(train_datasets)
    return fisher_dataset
     
def get_train_validation_dataloader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])
    rate = 0.2
    tedatalist = []
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)): 
        if i == args.validation_domain:
            tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=imgutil.image_train(args.dataset), test_envs=args.test_envs).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1-rate, random_state=args.seed)
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l*rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            tedatalist=ImageDataset(args.dataset, args.task, args.data_dir,
                                        names[i], i, transform=test_transform, indices=indexte, test_envs=args.test_envs)

    validation_loaders = DataLoader(
        dataset=tedatalist,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        pin_memory=True,
        shuffle=True)
    return validation_loaders