# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# import os
# SLICE_TYPE = '3g.40gb'
# SMI_LINE_ID = 1
# uuid = os.popen(f"nvidia-smi -L | sed -n 's/MIG {SLICE_TYPE}\(.*\): *//p' | sed -n '{SMI_LINE_ID}s/.$//p'").read()[2:-1]
# os.environ["CUDA_VISIBLE_DEVICES"] = uuid


import torch
import random
import numpy as np
import torchvision
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
from models.model import LimitNet
from tqdm import tqdm
from dahuffman import HuffmanCodec
import matplotlib.pyplot as plt  # Import matplotlib
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--image_path', type=str, default=None, help='Path to the image')
    parser.add_argument('--percentage', type=float, default=None, help='Percentage of the latent to keep')

    return parser.parse_args()


def sal_quantization_and_dequantization(data):
    min_ = torch.min(data)
    max_ = torch.max(data)
    data = (data - min_) / (max_ - min_)
    data = data * 255
    data = data.type(dtype=torch.uint8)

    data = data / 8
    data = data.type(dtype=torch.uint8)
    data = data * 8

    data = data / 255.0
    data = data * (max_ - min_) + min_
    return data

def quantization(data, filter_number, codec_setting):
    min_, max_ = codec_setting['min'][filter_number], codec_setting['max'][filter_number]
    data = (data - min_) / (max_ - min_)
    data = data * 255
    data = data.type(dtype=torch.uint8)
    
    quantization_step = 4
    data = data / quantization_step
    data = data.type(dtype=torch.uint8)

    return data

def quantization_and_dequantization(data, filter_number, codec_setting):
    min_, max_ = codec_setting['min'][filter_number], codec_setting['max'][filter_number]
    
    data = (data - min_) / (max_ - min_)
    data = data * 255
    data = data.type(dtype=torch.uint8)
    
    quantization_step = 4
    data = data / quantization_step
    data = data.type(dtype=torch.uint8)
    data = data * quantization_step

    data = data / 255.0
    data = data * (max_ - min_) + min_
    return data


                
def quantization_and_huffman(data, filter_number, codec_setting):
    data = data[data != -1].reshape(-1)
    
    if data.shape[0] == 0:
        return 0
    
    quantized_data = quantization(data, filter_number, codec_setting).cpu().numpy()
    codec = codec_setting['codec'][filter_number]
    encoded = codec.encode(quantized_data)
    return len(encoded) / 1024

def transformation(input_image):
    transform_test = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_test(input_image)
    
def save_sal(img, filename):
    resize_transform = transforms.Resize((224, 224), antialias=True)
    img_resized = resize_transform(img)[0].to('cpu').detach().numpy()
    img_resized = img_resized.squeeze()  
    plt.imsave(filename, img_resized, cmap='gray')


def save_decoded_image(img, filename):
    img = img[0]
    plt.imsave(filename, np.transpose(img, (1, 2, 0)))
    
def main():
    args = parse_args()

    model = LimitNet(args.model)
    model = torch.load(args.model_path)
    model.eval().to('cuda')
    model.p = args.percentage
    
    # read input image 
    input_image = Image.open(args.image_path).convert('RGB')
    input_image = transformation(input_image)
    input_image = input_image.unsqueeze(0).to('cuda')  # Add batch dimension and move to GPU

    encoded = model.encoder(input_image)
    saliency = model.sal_decoder(encoded)

    saliency = transforms.Resize((8, 8))(saliency)
    saliency = sal_quantization_and_dequantization(saliency)
    saliency = transforms.Resize((28, 28))(saliency)

    model.replace_tensor = torch.cuda.FloatTensor([0.0])[0]
    encoded = model.gradual_dropping(encoded, saliency)

    outputs = model.decoder(encoded)
    rec = outputs.clone()

    # plot_activations(bottleneck)
    save_sal(saliency, "saliency.png")
    save_decoded_image(rec.to('cpu').detach().numpy(), "ecoded_image.png")
    
if __name__ == "__main__":
    main()
