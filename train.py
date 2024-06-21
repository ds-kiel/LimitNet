# import os
# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# SLICE_TYPE = '3g.40gb'
# SMI_LINE_ID = 1
# uuid = os.popen(f"nvidia-smi -L | sed -n 's/MIG {SLICE_TYPE}\(.*\): *//p' | sed -n '{SMI_LINE_ID}s/.$//p'").read()[2:-1]
# os.environ["CUDA_VISIBLE_DEVICES"] = uuid

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import random
import numpy as np
import torchvision
import wandb
import pytorch_lightning as pl
import argparse
import cv2

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from models.model import LimitNet, CIFAR100Classifier
from models.data_preprocessing import CustomDataset, CIFAR100DataModule
from torchsummary import summary

wandb.require("core")

def set_seeds():
    torch.manual_seed(0)
    random.seed(10)
    np.random.seed(0)

def get_transforms():
    resize = 224
    imagenet_transform = transforms.Compose([
        transforms.Resize((resize, resize), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    SAL_imagenet_transform = transforms.Compose([
        transforms.Resize((resize, resize), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    SAL_saliancy_transform = transforms.Compose([
        transforms.Resize((28, 28), antialias=True),
    ])

    return imagenet_transform, SAL_imagenet_transform, SAL_saliancy_transform

def load_datasets(imagenet_root, imagenet_transform, SAL_imagenet_transform, SAL_saliancy_transform):
    ImageNet_data = datasets.ImageFolder(root=f'{imagenet_root}/train/', transform=imagenet_transform)
    ImageNet_train, _ = torch.utils.data.random_split(ImageNet_data, [50_000, len(ImageNet_data) - 50_000], generator=torch.Generator().manual_seed(41))
    ImageNet_val = datasets.ImageFolder(root=f'{imagenet_root}/val/', transform=imagenet_transform)

    ImageNetSal_train = CustomDataset(image_transform=SAL_imagenet_transform, saliancy_transform=SAL_saliancy_transform, train=True)
    ImageNetSal_val = CustomDataset(image_transform=SAL_imagenet_transform, saliancy_transform=SAL_saliancy_transform, train=False)

    return ImageNet_train, ImageNet_val, ImageNetSal_train, ImageNetSal_val

def train_phase_1(model, train_loader, val_loader, wandb_logger):
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=1,
        logger=wandb_logger
    )
    model.PHASE = 1
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("checkpoint/checkpoint_phase1.ckpt")

def train_phase_2(model, train_loader, val_loader, wandb_logger):
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=100,
        logger=wandb_logger
    )
    model.PHASE = 2
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("checkpoint/checkpoint_phase2.ckpt")

def train_phase_3(model, datamodule, wandb_logger, model_export_name):
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=20,
        logger=wandb_logger
    )
    model.PHASE = 3
    trainer.fit(model, datamodule=datamodule)
    torch.save(model.state_dict(), model_export_name)

def main(args):
    set_seeds()

    imagenet_transform, SAL_imagenet_transform, SAL_saliancy_transform = get_transforms()
    ImageNet_train, ImageNet_val, ImageNetSal_train, ImageNetSal_val = load_datasets(args.imagenet_root, imagenet_transform, SAL_imagenet_transform, SAL_saliancy_transform)

    model = LimitNet().to(device='cuda')
    model.PHASE = 3

    wandb_logger = WandbLogger(name=args.wandb_name, project=args.wandb_project)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Phase 1 Training
    train_loader = DataLoader(ImageNet_train, batch_size=args.batch_size, num_workers=8)
    val_loader = DataLoader(ImageNet_val, batch_size=args.batch_size, num_workers=8)
    train_phase_1(model, train_loader, val_loader, wandb_logger)

    # Phase 2 Training
    train_loader = DataLoader(ImageNetSal_train, batch_size=args.batch_size, num_workers=8)
    val_loader = DataLoader(ImageNetSal_val, batch_size=args.batch_size, num_workers=8)
    model = LimitNet.load_from_checkpoint(os.path.join(args.checkpoint_dir, "checkpoint_phase1.ckpt"))
    train_phase_2(model, train_loader, val_loader, wandb_logger)

    # Phase 3 Training
    if args.model=='cifar':
        model = LimitNet.load_from_checkpoint(os.path.join(args.checkpoint_dir, "checkpoint_phase2.ckpt"))
        model.cls_model = CIFAR100Classifier(learning_rate=0.001)
        model.cls_model.load_state_dict(torch.load(args.cifar_classifier_model_path))
    
        for param in model.sal_decoder.parameters():
            param.requires_grad = False
    
        datamodule = CIFAR100DataModule(batch_size=args.batch_size)
        train_phase_3(model, datamodule, wandb_logger, './LimitNet-CIFAR100-test')

    if args.model=='imagenet':
        model = LimitNet.load_from_checkpoint(os.path.join(args.checkpoint_dir, "checkpoint_phase2.ckpt"))
    
        for param in model.sal_decoder.parameters():
            param.requires_grad = False
    
        train_loader = torch.utils.data.DataLoader(ImageNet_train, batch_size=args.batch_size, num_workers=8)
        val_loader = torch.utils.data.DataLoader(ImageNet_val, batch_size=args.batch_size, num_workers=8)
        train_phase_3(model, datamodule, wandb_logger, './LimitNet-ImageNet-test')

    print("Training complete and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LimitNet model")
    parser.add_argument("--model", type=str, help="Dataset to train on")
    parser.add_argument("--batch_size", type=int,default=32, help="batch_size")
    parser.add_argument("--imagenet_root", type=str, default="/data22/datasets/ilsvrc2012/", help="ImageNet dataset root directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--wandb_name", type=str, default="LimitNet", help="WandB run name")
    parser.add_argument("--wandb_project", type=str, default="LimitNet", help="WandB project name")
    parser.add_argument("--cifar_classifier_model_path", type=str, default="./EfficentNet-CIFAR100-test", help="Path to the classifier model")

    args = parser.parse_args()
    main(args)
