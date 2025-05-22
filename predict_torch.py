"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw
import sys
import os
from pathlib import Path
import time
# import cv2  # Added for video processing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

import random

class_names = ['bus', 'bike', 'car', 'pedestrian', 'truck']

# Generate a fixed color map per class
random.seed(42)
label_colors = {
    name: tuple(random.randint(0, 255) for _ in range(3))
    for name in class_names
}


def draw(im, img_output_path, labels, boxes, scores, thrh=0.4):
    im = im.copy()
    draw = ImageDraw.Draw(im)

    keep = scores > thrh
    lab = labels[keep]
    box = boxes[keep]
    scrs = scores[keep]

    for j, b in enumerate(box):
        class_idx = lab[j].item()
        class_name = class_names[class_idx]
        color = label_colors.get(class_name, (255, 255, 255))  # fallback = white

        draw.rectangle(list(b.tolist()), outline=color, width=4)
        draw.text((b[0], b[1]), f"{class_name} {round(scrs[j].item(), 2)}", fill=color)

    im.save(img_output_path)


def process_image(model, device, file_path, visualize=True, visualize_dir='./visualize/test'):
    """Process a single image"""
    start = time.time()
    im_pil = load_image_fast(file_path)
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    im_data = transforms(im_pil).unsqueeze(0).to(device)
    print(f"Image loaded and transformed in {time.time() - start:.2f} seconds")
    output = model(im_data, orig_size)
    labels, boxes, scores = output
    if visualize:
        output_path = visualize_dir / file_path.name
        draw(im_pil, output_path, labels, boxes, scores)

def load_image_fast(file_path):
    img = cv2.imread(str(file_path))  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = Image.fromarray(img)
    return img

def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    file_paths = list(Path(args.input_dir).glob("*.png"))
    if len(file_paths) == 0:
        print("No .png files found in input directory.")
        return

    if args.visualize:
        Path(args.visualize_dir).mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    import time

    start_time = time.time()
    for file_path in tqdm(file_paths, desc="Processing Images"):
        process_image(model, device, Path(file_path), args.visualize, Path(args.visualize_dir))
    end_time = time.time()

    total_time = end_time - start_time
    fps = len(file_paths) / total_time if total_time > 0 else 0

    print(f"\nProcessed {len(file_paths)} images in {total_time:.2f} seconds. FPS: {fps:.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/deim_dfine/deim_hgnetv2_x_coco.yml')
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cuda:4')
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--visualize_dir', type=str, default='visualize/test')
    args = parser.parse_args()
    main(args)
