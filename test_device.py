#!/usr/bin/env python3
"""Проверка доступных ускорителей."""
import sys
print("Импорт torch... (может занять 10-20 сек при первом запуске)")
try:
    import torch
except Exception as e:
    print(f"[ОШИБКА] Не удалось импортировать torch: {e}")
    print("Возможно, torch установлен в системный Python, а не в venv.")
    print("Решение: удалите системный torch и установите внутри venv.")
    sys.exit(1)

print("=" * 50)
print("Проверка окружения AutoRotate")
print("=" * 50)
print(f"Python:    {sys.version.split()[0]}")
print(f"PyTorch:   {torch.__version__}")
print()

# CUDA
cuda = torch.cuda.is_available()
print(f"CUDA:      {'✅ Да' if cuda else '❌ Нет'}")
if cuda:
    print(f"  Устройство: {torch.cuda.get_device_name(0)}")
    print(f"  Память:     {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Intel XPU
xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()
print(f"Intel XPU: {'✅ Да' if xpu else '❌ Нет'}")
if xpu:
    try:
        print(f"  Устройство: {torch.xpu.get_device_name(0)}")
    except Exception:
        print(f"  Устройство: Intel XPU (детали недоступны)")

# Apple MPS
mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
print(f"Apple MPS: {'✅ Да' if mps else '❌ Нет'}")

# CPU
print(f"CPU:       ✅ Всегда доступен")
print()

# Итог
if cuda:
    dev = torch.device('cuda')
    print("✅ Будет использовано: NVIDIA CUDA")
elif xpu:
    dev = torch.device('xpu')
    print("✅ Будет использовано: Intel XPU (Arc 130V)")
elif mps:
    dev = torch.device('mps')
    print("✅ Будет использовано: Apple MPS")
else:
    dev = torch.device('cpu')
    print("⚠️  Будет использовано: CPU (медленно)")

# Тестовый прогон
try:
    x = torch.randn(2, 3, 224, 224).to(dev)
    y = x * 2
    print(f"\n✅ Тестовый тензор прошёл: {y.device}")
except Exception as e:
    print(f"\n❌ Ошибка тестового прогона: {e}")

print("=" * 50)
