# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import lightning as L
from models import CIFAR100DataModule, CIFAR100Classifier
import argparse
import torchvision


def main(args):
    datamodule = CIFAR100DataModule(batch_size=32)
    model = CIFAR100Classifier(learning_rate=0.001)
    print(type(model))
    trainer = L.Trainer(accelerator="gpu", max_epochs=100)
    trainer.fit(model, datamodule=datamodule)
    
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAR100 classifier")
    parser.add_argument("--save_path", type=str, default='./EfficentNet-CIFAR100', help="Path to save the model")
    args = parser.parse_args()
    main(args)
