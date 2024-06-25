# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



import os
SLICE_TYPE = '3g.40gb'
SMI_LINE_ID = 1
uuid = os.popen(f"nvidia-smi -L | sed -n 's/MIG {SLICE_TYPE}\(.*\): *//p' | sed -n '{SMI_LINE_ID}s/.$//p'").read()[2:-1]
os.environ["CUDA_VISIBLE_DEVICES"] = uuid

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.model import LimitNet  # Adjust this import based on your project structure
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--input_video', type=str, default=None, help='Path to the input video')
    parser.add_argument('--output_folder', type=str, default='./processed_frames/', help='Path to the output folder to save processed frames')
    parser.add_argument('--output_video', type=str, default='processed_video.mp4', help='Path to save the output video')
    parser.add_argument('--percentage', type=float, default=None, help='Percentage of the latent to keep')
    parser.add_argument('--frame_rate', type=int, default=30, help='Frame rate for the output video')

    return parser.parse_args()

def transformation(input_image):
    transform_test = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform_test(input_image)

def save_decoded_image(img, filename):
    img = img[0]
    plt.imsave(filename, np.transpose(img, (1, 2, 0)))

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

def extract_frames(video_path, frames_folder):
    os.makedirs(frames_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_folder, f'frame_{count:05d}.png')
        cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    return count

def process_movie_frames(model, model_path, input_folder, output_folder, output_video, percentage, frame_rate):
    os.makedirs(output_folder, exist_ok=True)
    
    model = LimitNet(model)
    model = torch.load(model_path)
    model.eval().to('cuda')
    model.p = percentage
    
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

    for frame_file in tqdm(frame_files, desc="Processing frames"):
        input_image_path = os.path.join(input_folder, frame_file)
        output_image_path = os.path.join(output_folder, frame_file)

        input_image = Image.open(input_image_path).convert('RGB')
        input_image = transformation(input_image)
        input_image = input_image.unsqueeze(0).to('cuda')

        encoded = model.encoder(input_image)
        saliency = model.sal_decoder(encoded)

        saliency = transforms.Resize((8, 8))(saliency)
        saliency = sal_quantization_and_dequantization(saliency)
        saliency = transforms.Resize((28, 28))(saliency)

        model.replace_tensor = torch.cuda.FloatTensor([0.0])[0]
        encoded = model.gradual_dropping(encoded, saliency)

        outputs = model.decoder(encoded)
        rec = outputs.clone()

        save_decoded_image(rec.to('cpu').detach().numpy(), output_image_path)
    
    frame_array = []
    for frame_file in frame_files:
        img = cv2.imread(os.path.join(output_folder, frame_file))
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, size)

    for frame in frame_array:
        out.write(frame)
    out.release()

if __name__ == "__main__":
    args = parse_args()
    
    frames_folder = './extracted_frames'
    frame_count = extract_frames(args.input_video, frames_folder)
    
    process_movie_frames(args.model, args.model_path, frames_folder, args.output_folder, args.output_video, args.percentage, args.frame_rate)
