# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from efficientnet_pytorch import EfficientNet
import pytorch_lightning as pl
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import cv2

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        datasets.CIFAR100(root='./data', train=True, download=True)
        datasets.CIFAR100(root='./data', train=False, download=True)

    def setup(self, stage=None):
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainset = datasets.CIFAR100(root='./data', train=True, transform=transform_train)
        self.testset = datasets.CIFAR100(root='./data', train=False, transform=transform_test)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, num_workers=2)
    

class CustomDataset(Dataset):
    def __init__(self, image_transform = None, saliancy_transform = None, train = None):
        self.train_image_list = sorted(os.listdir('/data22/aho/imagenet/train/image/'))
        self.train_saliancy_list = sorted(os.listdir('/data22/aho/imagenet/train/mask/'))
        # self.train_image_list = sorted(os.listdir('/data22/aho/basnet_data/train/image/'))
        # self.train_saliancy_list = sorted(os.listdir('/data22/aho/basnet_data/train/mask/'))
        
        self.train = train
        
        self.test_image_list = sorted(os.listdir('/data22/aho/imagenet/test/image/'))
        self.test_saliancy_list = sorted(os.listdir('/data22/aho/imagenet/test/mask/'))
        # self.test_image_list = sorted(os.listdir('/data22/aho/basnet_data/test/image/'))
        # self.test_saliancy_list = sorted(os.listdir('/data22/aho/basnet_data/test/mask/'))

        self.image_transform = image_transform
        self.saliancy_transform = saliancy_transform
        self.to_tensor = transforms.ToTensor()

        # with open('../BASNet/classes.pkl', 'rb') as f:
        # # with open('../../classes.pkl', 'rb') as f:
        #     self.classes = pickle.load(f)
        #     self.classes = dict((key+'.jpg', value) for (key, value) in self.classes.items())
            
    def __len__(self):
        if self.train:
            return len(self.train_image_list)
        else:
            return len(self.test_image_list)
        
    def __getitem__(self, index):
        if self.train:
            img_path = os.path.join('/data22/aho/imagenet/train/image/', self.train_image_list[index])
            saliancy_path = os.path.join('/data22/aho/imagenet/train/mask/', self.train_saliancy_list[index])
            # img_path = os.path.join('/data22/aho/basnet_data/train/image/', self.train_image_list[index])
            # saliancy_path = os.path.join('/data22/aho/basnet_data/train/mask/', self.train_saliancy_list[index])
        else:
            img_path = os.path.join('/data22/aho/imagenet/test/image/', self.test_image_list[index])
            saliancy_path = os.path.join('/data22/aho/imagenet/test/mask/', self.test_saliancy_list[index])
            # img_path = os.path.join('/data22/aho/basnet_data/test/image/', self.test_image_list[index])
            # saliancy_path = os.path.join('/data22/aho/basnet_data/test/mask/', self.test_saliancy_list[index])
            
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.to_tensor(image)
        
        
        saliancy = cv2.imread(saliancy_path, cv2.IMREAD_GRAYSCALE)
        saliancy = self.to_tensor(saliancy)
        
        if self.image_transform:
            image = self.image_transform(image)

        if self.saliancy_transform:
            saliancy = self.saliancy_transform(saliancy)

        # if self.train:
        #     label = self.classes[self.train_image_list[index]]
        # else:
        #     label = self.classes[self.test_image_list[index]]
        return (image, saliancy)