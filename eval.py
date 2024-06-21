# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# import os
# SLICE_TYPE = '3g.40gb'
# SMI_LINE_ID = 1
# uuid = os.popen(f"nvidia-smi -L | sed -n 's/MIG {SLICE_TYPE}\(.*\): *//p' | sed -n '{SMI_LINE_ID}s/.$//p'").read()[2:-1]
# os.environ["CUDA_VISIBLE_DEVICES"] = uuid

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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

def parse_args():
    parser = argparse.ArgumentParser(description="Model quantization and evaluation script")
    parser.add_argument("--model", type=str, help="Dataset to train evaludate")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--test_data_size', type=int, default=5000, help='Desired data size for testing')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument("--imagenet_root", type=str, default="/data22/datasets/ilsvrc2012/", help="ImageNet dataset root directory")
    return parser.parse_args()


def get_test_loader(args, est_data_size, resize, test_batch_size):
    transform_test = transforms.Compose([
        transforms.Resize((resize, resize), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.model == 'cifar':
        CIFAR_testset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform_test)
        return DataLoader(CIFAR_testset, batch_size=test_batch_size, num_workers=2)

    if args.model =='imagenet':
        ImageNet_val = datasets.ImageFolder(root=f'{args.imagenet_root}/val/', transform=transform_test)
        return torch.utils.data.DataLoader(IMGNET_test_set, batch_size=test_batch_size, num_workers=1)

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

def calculate_model_loss(model, k, test_loader, codec_setting):
    model.eval().to('cuda')
    size_list, acc_list = [], []
    for data in test_loader:
        images, labels = data
        images = images.to('cuda')
        encoded = model.encoder(images)
        saliency = model.sal_decoder(encoded)
        saliency = transforms.Resize((8, 8))(saliency)
        saliency = sal_quantization_and_dequantization(saliency)
        saliency = transforms.Resize((28, 28))(saliency)
        # to correctly calculate the encoded data size, we set the valus which are originaly zero to -1 to diffrentiate between them and the values which we want to drop.
        model.replace_tensor = torch.cuda.FloatTensor([-1.0])[0] 
        bottleneck = model.gradual_dropping(encoded, saliency).detach().clone()

        # after creating the correct tensor for encoded data, we set the droped values to zero for decoding
        model.replace_tensor = torch.cuda.FloatTensor([0.0])[0]
        encoded = model.gradual_dropping(encoded, saliency)
        
        for i in range(encoded.size(0)):
            for j in range(encoded.size(1)):
                encoded[i, j] = quantization_and_dequantization(encoded[i, j], j, codec_setting)
                
        outputs = model.decoder(encoded)
        outputs = model.transforms(outputs)
        rec = outputs.clone()
        outputs = model.cls_model(outputs)
        acc = top_k_acc(outputs, labels, k)
        acc_list.append(acc)
        
        for image in bottleneck:
            map_size_list = []
            for feature_map in range(image.size(0)):
                data_size = quantization_and_huffman(image[feature_map], feature_map, codec_setting)
                map_size_list.append(data_size)
            size_list.append(np.sum(map_size_list))
            
    return np.mean(size_list), np.mean(acc_list)

def top_k_acc(outputs, labels, k):
    outputs = torch.softmax(outputs, dim=1)
    preds = torch.topk(outputs, k=k).indices.squeeze(0).tolist()
    batch_acc = sum([1 for p, l in zip(preds, labels) if l in p])
    return batch_acc / len(labels)

def create_codec(test_loader, model):
    codec_setting = {
        'min': {},
        'max': {},
        'codec': {}
    }
    temp_loader = DataLoader(test_loader.dataset, batch_size=5000, num_workers=1)
    images, labels = next(iter(temp_loader))
    images = images.to('cuda')

    for i in tqdm(range(12)):
        encoded = model.encoder(images)        
        data = encoded[:, i, :, :].reshape(-1).detach().clone()
        min_, max_ = torch.min(data), torch.max(data)
        data = ((data - min_) / (max_ - min_) * 255).type(torch.uint8)
        data = (data / 4).type(torch.uint8).cpu().numpy()
        
        # adding dummy data for covering all of the possible values
        for j in range(0,63):
            if j not in data:
                data = np.append(data,j)
        
        codec = HuffmanCodec.from_data(data)
        codec_setting['min'][i], codec_setting['max'][i], codec_setting['codec'][i] = min_, max_, codec
        
    return codec_setting

def main():
    args = parse_args()
    torch.manual_seed(0)
    random.seed(10)
    np.random.seed(0)
    
    model = LimitNet().to('cuda')
    model.load_state_dict(torch.load(args.model_path))
    
    test_loader = get_test_loader(args, args.test_data_size, 224, args.test_batch_size)
    
    codec_setting = create_codec(test_loader, model)

    top1 = []
    sizes = []
    for threshold in range(1, 101, 7):
        model.p = threshold / 100
        images_size, total_acc = calculate_model_loss(model,
                                                      k=1,
                                                      test_loader=test_loader,
                                                      codec_setting=codec_setting
                                                      )
        sizes.append(images_size)
        top1.append(total_acc)
        print(f'Threshold: {model.p:.2f}, Image Size (KB): {images_size:.2f}, Model Top 1 Accuracy: {100*total_acc:.2f}')
    

    # Plotting Image Size vs Accuracy
    dataset_type = 'ImageNet1K' if args.model == 'imagenet' else 'CIFAR-100'
    plt.figure(figsize=(12, 8))

    # Plot with enhanced aesthetics
    plt.plot(sizes, [100 * acc for acc in top1], marker='o', linestyle='-', color='b', markersize=8, linewidth=2, label=f'{dataset_type}')

    # Adding labels and title
    plt.xlabel('Image Size (KB)', fontsize=14)
    plt.ylabel('Model Top 1 Accuracy (%)', fontsize=14)
    plt.title(f'Image Size vs Accuracy for {dataset_type}', fontsize=16)

    # Adding grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)

    # Enhancing ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save the plot
    plt.savefig(f'image_size_vs_accuracy_{dataset_type.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    main()
