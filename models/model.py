# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import pytorch_lightning as pl
from copy import copy
import torchvision
import random
from dahuffman import HuffmanCodec
import numpy as np
import cv2
import torch
import json
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torchmetrics import Accuracy
from torch.utils.data import Dataset
import os
import pickle
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb
import torchvision.models as models

device = 'cuda'

class CIFAR100Classifier(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=100)
        self.accuracy = Accuracy(task="multiclass", num_classes=100)
        
        # # Freeze the backbone
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Modify the classifier head
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, 100)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        outputs = self(images)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return {
            'loss': loss
        }

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        outputs = self(images)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)

        return {
            'loss': loss,
            'acc': acc
        }

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer



#Define the Convolutional Autoencoder

class Encoder(pl.LightningModule):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # Encoder
        self.run_on_nRF = False
        self.conv1 = nn.Conv2d(3, 16, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=4, padding=1)
        self.conv3 = nn.Conv2d(16, 12, 3, stride=1, padding=1)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

 # Define the Convolutional Autoencoder


class Decoder(pl.LightningModule):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Encoder
        self.t_conv1 = nn.ConvTranspose2d(12, 64, 7, stride=2, padding=3, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(64, 64, 5, stride=4, padding=1, output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.t_conv4 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.t_conv5 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)
        # self.enhancment = Enhancement()

        
    def forward(self, x):
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x)) 
        x = F.relu(self.t_conv3(x)) 
        x = F.relu(self.t_conv4(x)) 
        x = self.t_conv5(x)
        x = torch.sigmoid(x)
        return x



    
# Define the Convolutional Autoencoder

class SalDecoder(pl.LightningModule):
    def __init__(self):
        super(SalDecoder, self).__init__()
       
        # Encoder
        self.sal_t_conv1 = nn.Conv2d(12, 16, 3, stride=1, padding=1)
        self.sal_t_conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.sal_t_conv3 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.sal_t_conv4 = nn.Conv2d(8, 4, 3, stride=1, padding=1)
        self.sal_t_conv5 = nn.Conv2d(4, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.sal_t_conv1(x))
        x = F.relu(self.sal_t_conv2(x))
        x = F.relu(self.sal_t_conv3(x))
        x = F.relu(self.sal_t_conv4(x))
        x = torch.sigmoid(self.sal_t_conv5(x))
        return x
   

def random_noise(self, x, r1, r2):
    temp_x = x.clone()
    noise = (r1 - r2) * torch.rand(x.shape) + r2
    return torch.clamp(temp_x + noise, min=0.0, max=1.0)




#Define the Convolutional Autoencoder

class LimitNet(pl.LightningModule):
    def __init__(self):
        super(LimitNet, self).__init__()
        #
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.sal_decoder = SalDecoder()
        ###
        
        self.accuracy = Accuracy(task="multiclass", num_classes=100)
        self.cls_model = CIFAR100Classifier(learning_rate=0.001)
        # self.cls_model = torch.load('./EfficentNet')

        self.transforms = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.PHASE = None
        self.p = None
        self.replace_tensor = torch.cuda.FloatTensor([0.0])[0]

        self.training_step_loss = []
        self.training_step_loss_saliency = []
        self.training_step_loss_ce = []
        self.training_step_loss_rec = []
        self.training_step_acc = []

        self.validation_step_loss = []
        self.validation_step_loss_saliency = []
        self.validation_step_loss_ce = []
        self.validation_step_loss_rec = []
        self.validation_step_acc = []
        
    def random_noise(self, x, r1, r2):
        temp_x = x.clone()
        noise = (r1 - r2) * torch.rand(x.shape) + r2
        return torch.clamp(temp_x + noise.cuda(), min=0.0, max=1.0)
    

    def forward(self, x):
        
        if self.PHASE == 1:
            
            # Encoder
            x = self.encoder(x)
            x = self.rate_less(x)

            # Decoder
            # x = self.random_noise(x, -0.001, 0.001)
            noise = torch.rand_like(x, dtype=torch.float) * 0.02 - 0.01
            x = x + noise
            
            x = self.transforms(self.decoder(x))
            self.rec_image = x.detach().clone()

            return x
        
        if self.PHASE == 2:
            
            # Encoder
            x = self.encoder(x)
            
            sal = self.sal_decoder(x)
            self.sal = sal.detach().clone()

            x = self.gradual_dropping(x, sal)

            # Decoder
            # x = self.random_noise(x, -0.001, 0.001)
            noise = torch.rand_like(x, dtype=torch.float) * 0.02 - 0.01
            x = x + noise
            
            x = self.transforms(self.decoder(x))
            
            self.rec_image = x.detach().clone()
            
            return x, sal

            
        # adding classification to model
        if self.PHASE == 3:
            
            # Encoder
            x = self.encoder(x)

            sal = self.sal_decoder(x)
            x = self.gradual_dropping(x, sal)
            
            # Decoder
            # x = self.random_noise(x, -0.001, 0.001)
            noise = torch.rand_like(x, dtype=torch.float) * 0.02 - 0.01
            x = x + noise
            
            x = self.transforms(self.decoder(x))

            self.rec_image = x.detach().clone()
            
            x = self.cls_model(x)
            
            return x

    
    def configure_optimizers(self):
        
        if self.PHASE == 1:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

            return {
               'optimizer': optimizer,
           }
        
        if self.PHASE == 2:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            
            return {
               'optimizer': optimizer,
           }
        
        if self.PHASE == 3:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.00005)
            
            return {
               'optimizer': optimizer,
           } 

    def training_step(self, train_batch, batch_idx):
        
        if self.PHASE == 1:
            images, _ = train_batch
            outputs = self(images)
            loss = nn.MSELoss()(outputs, images)
            self.log('during_train_loss_mse_phase1', loss, on_epoch=True, prog_bar=True, logger=True)
            
            self.training_step_loss.append(loss)
            self.training_step_loss_rec.append(loss)

            return {
                'loss': loss
            }
        
        if self.PHASE == 2:
            images, saliency= train_batch
            outputs, outputs_sal = self(images)
            
            loss = nn.MSELoss()(outputs, images)
            loss_saliency = nn.BCELoss()(outputs_sal, saliency)

            self.log('during_train_loss_mse_phase2', loss, on_epoch=True, prog_bar=True, logger=True)
            self.log('during_train_loss_saliency_phase2', loss_saliency, on_epoch=True, prog_bar=True, logger=True)

            self.training_step_loss.append(1 * loss + 0.05 * loss_saliency)
            self.training_step_loss_saliency.append(loss_saliency)
            self.training_step_loss_rec.append(loss)

            return {
                'loss': 1 * loss + 0.05 * loss_saliency,
                     }

        if self.PHASE == 3:
            images , labels = train_batch
            outputs = self(images)
            loss_ce = nn.CrossEntropyLoss()(outputs, labels)

            self.log('during_train_loss_CE_phase3', loss_ce, on_epoch=True, prog_bar=True, logger=True)
            preds = torch.argmax(outputs, dim=1)
            acc = self.accuracy(preds, labels)
            self.log('during_train_acc_phase3', acc, on_epoch=True, prog_bar=True, logger=True)
    
            self.training_step_loss_ce.append(loss_ce)
            self.training_step_acc.append(acc)
            self.training_step_loss.append(loss_ce)

            return {
                'loss': loss_ce ,
                   }

    def on_train_epoch_end(self):
        if self.PHASE == 1:
            loss_rec = torch.stack([x for x in self.training_step_loss_rec]).mean()
            self.log('train_loss_rec_epoch_phase1', loss_rec, on_epoch=True, prog_bar=True, logger=True)

        if self.PHASE == 2:

            loss_rec = torch.stack([x for x in self.training_step_loss_rec]).mean()
            self.log('train_loss_rec_epoch_phase2', loss_rec, on_epoch=True, prog_bar=True, logger=True)

            loss_saliency = torch.stack([x for x in self.training_step_loss_saliency]).mean()
            self.log('train_loss_saliency_epoch_phase2', loss_saliency, on_epoch=True, prog_bar=True, logger=True)

        if self.PHASE == 3:

            acc = torch.stack([x for x in self.training_step_acc]).mean()
            loss_ce = torch.stack([x for x in self.training_step_loss_ce]).mean()

            self.log('train_loss_CE_epoch_phase3', loss_ce, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_acc_epoch_phase3', acc, on_epoch=True, prog_bar=True, logger=True)

        self.training_step_loss.clear()
        self.training_step_loss_saliency.clear()
        self.training_step_loss_ce.clear()
        self.training_step_loss_rec.clear()
        self.training_step_acc.clear()

    def validation_step(self, val_batch, batch_idx):
        loss_list = []
        loss_saliency_list = []
        acc_list = []
        loss_ce_list = []
        
        if self.PHASE == 1:
            for t in [0.2]:
                self.p = t
                images, _ = val_batch
                outputs = self(images)
            
                loss = nn.MSELoss()(outputs, images)
                loss_list.append(loss.item())
                
            loss = torch.Tensor([np.mean(loss_list)])
            self.p = None

            self.validation_step_loss.append(loss)
            self.validation_step_loss_rec.append(loss)
            return {'loss': loss }
        
        if self.PHASE == 2:
            for t in [0.2]:
                self.p = t
                images, saliency= val_batch
                outputs, outputs_sal = self(images)
                loss = nn.MSELoss()(outputs, images)
                loss_saliency = nn.BCELoss()(outputs_sal, saliency)
                
                loss_list.append(loss.item())
                loss_saliency_list.append(loss_saliency.item())
                
            loss = torch.Tensor([np.mean(loss_list)])
            loss_saliency = torch.Tensor([np.mean(loss_saliency_list)])
            self.p = None
            
            self.validation_step_loss.append(1 * loss + 0.05 * loss_saliency)
            self.validation_step_loss_rec.append(loss)
            self.validation_step_loss_saliency.append(loss_saliency)

            return {'loss': 1 * loss + 0.05 * loss_saliency}
        
        if self.PHASE == 3:
            for t in [0.2]:
                self.p = t
                images , labels = val_batch
                outputs = self(images)
                loss_ce = nn.CrossEntropyLoss()(outputs, labels)

                preds = torch.argmax(outputs, dim=1)
                acc = self.accuracy(preds, labels)
                
                loss_ce_list.append(loss_ce.item())
                acc_list.append(acc.item())
                
            loss_ce = torch.Tensor([np.mean(loss_ce_list)])
            acc = torch.Tensor([np.mean(acc_list)])
            
            self.p = None
            self.validation_step_loss.append(loss_ce)
            self.validation_step_loss_ce.append(loss_ce)
            self.validation_step_acc.append(acc)
            return {
                    'loss_ce': loss_ce ,
                    'acc': acc,
                   }

    def on_validation_epoch_end(self):

        if self.PHASE == 1:
            
            loss_rec = torch.stack([x for x in self.validation_step_loss_rec]).mean()
            self.log('val_loss_rec_epoch_phase1', loss_rec, on_epoch=True, prog_bar=True, logger=True)

            self.logger.experiment.log({"rec_image_phase1": wandb.Image(self.rec_image[0])})

            
        if self.PHASE == 2:
            
            loss_rec = torch.stack([x for x in self.validation_step_loss_rec]).mean()
            self.log('val_loss_rec_epoch_phase2', loss_rec, on_epoch=True, prog_bar=True, logger=True)

            loss_saliency = torch.stack([x for x in self.validation_step_loss_saliency]).mean()
            self.log('val_loss_saliency_epoch_phase2', loss_saliency, on_epoch=True, prog_bar=True, logger=True)

            self.logger.experiment.log({"rec_image_phase2": wandb.Image(self.rec_image[0])})
            self.logger.experiment.log({"sal_phase2": wandb.Image(self.sal[0])})

            
        if self.PHASE == 3:
            
            acc = torch.stack([x for x in self.validation_step_acc]).mean()
            loss_ce = torch.stack([x for x in self.validation_step_loss_ce]).mean()
            
            self.logger.experiment.log({"rec_image_phase3": wandb.Image(self.rec_image[0])})

            self.log('val_acc_epoch_phase3', acc, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss_CE_epoch_phase3', loss_ce, on_epoch=True, prog_bar=True, logger=True)
            
        self.p = None
        self.validation_step_loss.clear()
        self.validation_step_loss_saliency.clear()
        self.validation_step_loss_ce.clear()
        self.validation_step_loss_rec.clear()
        self.validation_step_acc.clear()
        
    def gradual_dropping(self,x, sal):
        temp_x = x.clone()
        
        for i in range(x.shape[0]):
            
            saliancy = sal[i].repeat(12, 1, 1)
            
            for j in range(saliancy.shape[0]):
                saliancy[j,:,:] = saliancy[j,:,:] + (12-j)/ 5
                # saliancy[j,:,:] = saliancy[j,:,:] + 1

            if self.p:
                # p shows the percentage of keeping
                p = self.p
            else:
                p = np.random.uniform(0,1.0,1)[0]                
            if p != 1.0:            
                q = torch.quantile(saliancy.view(-1), 1-p, dim=0, keepdim=True)
                selection = (saliancy < q) # selection for droping
                selection = selection.view(12, 28, 28)
                temp_x[i,:,:,:] = torch.where(
                    selection,
                    self.replace_tensor,
                    x[i,:,:,:],
                )
                
        return temp_x
    
    def rate_less(self,x):
        
        temp_x = x.clone()
        for i in range(x.shape[0]):
            if self.p:
                # p shows the percentage of keeping
                p = self.p
            else:
                p = np.random.uniform(0,1.0,1)[0]                
            if p != 1.0:            
                p = int(p * x.shape[1])
                replace_tensor = torch.rand(x.shape[1]-p, x.shape[2], x.shape[3]).fill_(0)
                temp_x[i,-(x.shape[1]-p):,:,:] =  replace_tensor
                
        return temp_x
