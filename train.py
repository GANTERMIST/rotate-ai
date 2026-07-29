#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoRotate — локальное обучение классификации ориентации изображений.
Адаптировано для: Intel Arc 130V, NVIDIA CUDA, Apple MPS, CPU.
Все артефакты сохраняются в директории проекта.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision import datasets, models
import timm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# 1. Утилиты
# ═══════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Автоопределение: NVIDIA CUDA -> Intel XPU -> Apple MPS -> CPU"""
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f"[Устройство] NVIDIA CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        dev = torch.device('xpu')
        try:
            print(f"[Устройство] Intel XPU: {torch.xpu.get_device_name(0)}")
        except Exception:
            print("[Устройство] Intel XPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        dev = torch.device('mps')
        print("[Устройство] Apple MPS")
    else:
        dev = torch.device('cpu')
        print("[Устройство] CPU")
    return dev

def ensure_dirs(base_dir):
    dirs = {
        'data': os.path.join(base_dir, 'data'),
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'plots': os.path.join(base_dir, 'plots'),
        'results': os.path.join(base_dir, 'results'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

# ═══════════════════════════════════════════════════════════════
# 2. Датасет
# ═══════════════════════════════════════════════════════════════

class RotationDataset(Dataset):
    def __init__(self, base_dataset, img_size=224, seed=None):
        self.base = base_dataset
        self.img_size = img_size
        self.angles = [0, 90, 180, 270]
        self.seed = seed
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        if self.seed is not None:
            angle_label = random.Random(self.seed + idx).randint(0, 3)
        else:
            angle_label = random.randint(0, 3)
        angle = self.angles[angle_label]
        img_rotated = img.rotate(angle, expand=True)
        img_tensor = self.transform(img_rotated)
        return img_tensor, angle_label

# ═══════════════════════════════════════════════════════════════
# 3. Модели
# ═══════════════════════════════════════════════════════════════

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def build_efficientnet(num_classes=4, pretrained=True):
    try:
        model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
    except Exception as e:
        print(f"[WARN] Не удалось загрузить pretrained EfficientNet: {e}")
        print("[WARN] Создаём с random weights...")
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    return model

def build_resnet(num_classes=4, pretrained=True):
    try:
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ═══════════════════════════════════════════════════════════════
# 4. Обучение / валидация
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False, ncols=100)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Eval", leave=False, ncols=100)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels

# ═══════════════════════════════════════════════════════════════
# 5. Визуализация
# ═══════════════════════════════════════════════════════════════

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def save_samples(dataset, save_path, n=8):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    angle_names = ['0°', '90°', '180°', '270°']
    for row in range(2):
        for col in range(4):
            idx = row * 4 + col
            if idx >= n:
                break
            img_tensor, label = dataset[idx]
            img = denormalize(img_tensor).permute(1, 2, 0).clamp(0, 1).numpy()
            axes[row, col].imshow(img)
            axes[row, col].set_title(angle_names[label], fontsize=11)
            axes[row, col].axis('off')
    plt.suptitle("Примеры изображений с разными углами поворота", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Сохранено] Примеры данных: {save_path}")

def save_training_curves(histories, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for hist, name in histories:
        axes[0].plot(hist['val_acc'], marker='o', label=name)
    axes[0].set_title('Validation Accuracy')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    for hist, name in histories:
        axes[1].plot(hist['val_loss'], marker='o', label=name)
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Сохранено] Графики обучения: {save_path}")

def save_confusion_matrix(labels, preds, class_names, save_path, title):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix — {title}')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Сохранено] Матрица ошибок: {save_path}")

def save_error_examples(model, dataset, device, save_path, n=8):
    model.eval()
    errors = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    angle_names = ['0°', '90°', '180°', '270°']

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            if len(errors) >= n:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1).item()
            if pred != y.item():
                errors.append((x.cpu(), y.item(), pred))

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for ax, (img, true, pred) in zip(axes, errors):
        img = denormalize(img.squeeze()).permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.set_title(f"Истина: {angle_names[true]}\nПредсказано: {angle_names[pred]}", color='red')
        ax.axis('off')
    for ax in axes[len(errors):]:
        ax.axis('off')
    plt.suptitle("Примеры ошибок модели", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Сохранено] Примеры ошибок: {save_path}")

def save_demo(rotator, cifar_test, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i in range(4):
        idx = i * 500
        cifar_img = cifar_test[idx][0]
        rng = random.Random(idx)
        angle_label = rng.randint(0, 3)
        angle = [0, 90, 180, 270][angle_label]
        rotated_img = cifar_img.rotate(angle, expand=True)
        corrected, pred_angle = rotator.correct_orientation(rotated_img)
        axes[0, i].imshow(rotated_img)
        axes[0, i].set_title(f"Вход: {angle}°")
        axes[0, i].axis('off')
        axes[1, i].imshow(corrected)
        axes[1, i].set_title(f"Выход: {pred_angle}°")
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Исходное", fontsize=12)
    axes[1, 0].set_ylabel("Исправленное", fontsize=12)
    plt.suptitle("AutoRotate: коррекция ориентации", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Сохранено] Демо инференса: {save_path}")

# ═══════════════════════════════════════════════════════════════
# 6. Инференс-класс
# ═══════════════════════════════════════════════════════════════

class AutoRotator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.angles = [0, 90, 180, 270]
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

    def predict_angle(self, image_pil):
        tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(tensor).argmax(1).item()
        return self.angles[pred], pred

    def correct_orientation(self, image_pil):
        angle, _ = self.predict_angle(image_pil)
        if angle == 0:
            return image_pil, angle
        return image_pil.rotate(-angle, expand=True), angle

# ═══════════════════════════════════════════════════════════════
# 7. Пайплайн эксперимента
# ═══════════════════════════════════════════════════════════════

def run_experiment(model, model_name, train_loader, val_loader, test_loader,
                   device, epochs, lr, weight_decay, dirs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    best_val_acc = 0.0
    best_state = None

    print(f"\n{'='*60}")
    print(f"ЭКСПЕРИМЕНТ: {model_name}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    _, test_acc, preds, labels = eval_epoch(model, test_loader, criterion, device)
    f1 = f1_score(labels, preds, average='macro')

    print(f"Test Accuracy: {test_acc:.4f} | F1-macro: {f1:.4f}")

    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    ckpt_path = os.path.join(dirs['checkpoints'], f"{safe_name}.pth")
    arch = 'efficientnet_b0' if 'efficientnet' in model_name.lower() else 'resnet18' if 'resnet' in model_name.lower() else 'baseline'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'model': model_name, 'arch': arch, 'classes': 4, 'img_size': 224},
        'metrics': {'accuracy': float(test_acc), 'f1': float(f1)}
    }, ckpt_path)
    print(f"[Сохранено] Чекпоинт: {ckpt_path}")

    return {
        'model': model,
        'name': model_name,
        'history': history,
        'test_acc': test_acc,
        'f1': f1,
        'preds': preds,
        'labels': labels,
        'params': sum(p.numel() for p in model.parameters())
    }

# ═══════════════════════════════════════════════════════════════
# 8. main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='AutoRotate — локальное обучение')
    parser.add_argument('--epochs', type=int, default=10, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=64, help='Размер батча')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Размер изображения')
    parser.add_argument('--workers', type=int, default=None, help='Workers DataLoader (auto: 0 для Windows)')
    parser.add_argument('--models', type=str, default='all', help='all | baseline | efficientnet_freeze | efficientnet_full | resnet')
    parser.add_argument('--skip_train', action='store_true', help='Только подготовка данных, без обучения')
    args = parser.parse_args()

    if args.workers is None:
        args.workers = 0 if sys.platform == 'win32' else 4

    set_seed(42)
    device = get_device()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = ensure_dirs(base_dir)

    print(f"\n[Конфиг] epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, workers={args.workers}")
    print(f"[Директории] {dirs}")

    # ─── Данные ───
    print("\nЗагрузка CIFAR-10...")
    cifar_train = datasets.CIFAR10(root=dirs['data'], train=True, download=True)
    cifar_test = datasets.CIFAR10(root=dirs['data'], train=False, download=True)

    train_ds_full = RotationDataset(cifar_train, img_size=args.img_size, seed=None)
    test_ds = RotationDataset(cifar_test, img_size=args.img_size, seed=42)

    train_size = int(0.85 * len(train_ds_full))
    val_size = len(train_ds_full) - train_size
    train_ds, val_ds = random_split(
        train_ds_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    pin = device.type in ('cuda', 'xpu')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    save_samples(train_ds_full, os.path.join(dirs['plots'], '01_samples.png'))

    if args.skip_train:
        print("Обучение пропущено (--skip_train).")
        return

    # ─── Эксперименты ───
    results = []
    histories = []

    models_to_run = []
    if args.models in ('all', 'baseline'):
        models_to_run.append(('Baseline CNN', BaselineCNN().to(device), args.lr))
    if args.models in ('all', 'efficientnet_freeze'):
        m = build_efficientnet().to(device)
        for p in m.parameters():
            p.requires_grad = False
        for p in m.get_classifier().parameters():
            p.requires_grad = True
        models_to_run.append(('EfficientNet-B0 (freeze)', m, 1e-3))
    if args.models in ('all', 'efficientnet_full'):
        models_to_run.append(('EfficientNet-B0 (full)', build_efficientnet().to(device), args.lr))
    if args.models in ('all', 'resnet'):
        models_to_run.append(('ResNet-18 (full)', build_resnet().to(device), args.lr))

    for name, model, lr in models_to_run:
        res = run_experiment(
            model, name, train_loader, val_loader, test_loader,
            device, args.epochs, lr, 1e-4, dirs
        )
        results.append(res)
        histories.append((res['history'], name))

    # ─── Сводная таблица ───
    print("\n" + "=" * 70)
    print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 70)
    print(f"{'Модель':<30} {'Test Acc':>10} {'F1-macro':>10} {'Параметров':>15}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30} {r['test_acc']:>10.4f} {r['f1']:>10.4f} {r['params']:>15,}")

    # ─── Графики ───
    save_training_curves(histories, os.path.join(dirs['plots'], '02_training_curves.png'))

    # ─── Лучшая модель ───
    best = max(results, key=lambda x: x['test_acc'])
    print(f"\nЛучшая модель: {best['name']} (accuracy={best['test_acc']:.4f})")

    save_confusion_matrix(
        best['labels'], best['preds'],
        ['0°', '90°', '180°', '270°'],
        os.path.join(dirs['plots'], '03_confusion_matrix.png'),
        best['name']
    )

    print("\nClassification Report (лучшая модель):")
    print(classification_report(best['labels'], best['preds'], target_names=['0°', '90°', '180°', '270°']))

    # ─── Ошибки ───
    save_error_examples(
        best['model'], test_ds, device,
        os.path.join(dirs['plots'], '04_errors.png')
    )

    # ─── Демо инференса ───
    rotator = AutoRotator(best['model'], device)
    save_demo(rotator, cifar_test, os.path.join(dirs['plots'], '05_demo.png'))

    # ─── Метрики в JSON ───
    metrics_path = os.path.join(dirs['results'], 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'config': vars(args),
            'results': [
                {'model': r['name'], 'accuracy': float(r['test_acc']), 'f1': float(r['f1']), 'params': r['params']}
                for r in results
            ],
            'best_model': best['name']
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[Сохранено] Метрики: {metrics_path}")

    print("\n" + "=" * 60)
    print("ГОТОВО. Все файлы сохранены в директории проекта:")
    for k, v in dirs.items():
        print(f"  {k}: {v}")
    print("=" * 60)

if __name__ == '__main__':
    main()
