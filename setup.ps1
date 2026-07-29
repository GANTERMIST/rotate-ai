Write-Host "=========================================="
Write-Host " AutoRotate — установка зависимостей"
Write-Host "=========================================="
Write-Host ""

python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ОШИБКА] Python не найден. Установите Python 3.11 с https://python.org"
    Read-Host "Нажмите Enter для выхода"
    exit 1
}

Write-Host "[1/3] Обновление pip..."
python -m pip install --upgrade pip

Write-Host ""
Write-Host "[2/3] Установка PyTorch + XPU (Intel Arc)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

Write-Host ""
Write-Host "[3/3] Установка остальных библиотек..."
pip install -r requirements.txt

Write-Host ""
Write-Host "=========================================="
Write-Host " Установка завершена!"
Write-Host " Проверьте устройство: python test_device.py"
Write-Host "=========================================="
Read-Host "Нажмите Enter для выхода"
