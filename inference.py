#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoRotate — инференс на локальных изображениях.
Берёт последний (или указанный) чекпоинт из папки checkpoints/.
"""

import os
import sys
import argparse
import torch
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from train import AutoRotator, get_device, build_efficientnet, build_resnet

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    config = ckpt.get('config', {})
    arch = config.get('arch', 'efficientnet_b0')
    num_classes = config.get('classes', 4)

    if arch == 'efficientnet_b0':
        model = build_efficientnet(num_classes=num_classes, pretrained=False)
    elif arch == 'resnet18':
        model = build_resnet(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Неизвестная архитектура: {arch}")

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config

def main():
    parser = argparse.ArgumentParser(description='AutoRotate Inference')
    parser.add_argument('--input', '-i', required=True, help='Путь к входному изображению')
    parser.add_argument('--output', '-o', default=None, help='Куда сохранить (по умолчанию перезаписывает input)')
    parser.add_argument('--ckpt', '-c', default=None, help='Путь к .pth (авто: последний из ./checkpoints/)')
    args = parser.parse_args()

    device = get_device()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Автопоиск чекпоинта
    if args.ckpt is None:
        ckpt_dir = os.path.join(base_dir, 'checkpoints')
        if os.path.isdir(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
            if not ckpts:
                print("Ошибка: не найдено чекпоинтов в ./checkpoints/")
                print("Сначала запустите обучение: python train.py")
                sys.exit(1)
            args.ckpt = os.path.join(ckpt_dir, max(ckpts, key=lambda x: os.path.getctime(os.path.join(ckpt_dir, x))))
            print(f"Автовыбор чекпоинта: {args.ckpt}")
        else:
            print("Ошибка: укажите --ckpt или создайте ./checkpoints/")
            sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Ошибка: файл не найден: {args.input}")
        sys.exit(1)

    print(f"Загрузка модели...")
    model, config = load_model(args.ckpt, device)
    rotator = AutoRotator(model, device)

    img = Image.open(args.input).convert('RGB')
    corrected, angle = rotator.correct_orientation(img)

    out_path = args.output if args.output else args.input
    corrected.save(out_path)
    print(f"Определён угол: {angle}°")
    print(f"Результат сохранён: {out_path}")

if __name__ == '__main__':
    main()
