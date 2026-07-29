@echo off
chcp 65001 >nul
echo ==========================================
echo  AutoRotate — установка зависимостей
echo ==========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден. Установите Python 3.11 с https://python.org
    pause
    exit /b 1
)

echo [1/3] Python найден:
python --version
echo.

echo [2/3] Обновление pip...
python -m pip install --upgrade pip

echo.
echo [3/3] Установка PyTorch + XPU (Intel Arc)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

echo.
echo [4/4] Установка остальных библиотек...
pip install -r requirements.txt

echo.
echo ==========================================
echo  Установка завершена!
echo  Проверьте устройство: python test_device.py
echo ==========================================
pause
