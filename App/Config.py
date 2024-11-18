import os
import sys

# Получаем путь к текущему скрипту
if getattr(sys, 'frozen', False):  # Если приложение скомпилировано
    ROOT_DIR = os.path.dirname(sys.executable)  # Путь к каталогу с .exe
else:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Путь к каталогу с исходным скриптом

# Разделяем путь в зависимости от операционной системы
if os.name == 'nt':
    ROOT_DIR = ROOT_DIR.replace('/', '\\')  # Для Windows используем обратный слэш
elif os.name == 'posix':
    ROOT_DIR = ROOT_DIR.replace('\\', '/')  # Для Linux/Mac используем прямой слэш
else:
    raise NotImplementedError(f"Unsupported OS: {os.name}")

# Полные пути к моделям и другим файлам
YOLO_PATH = os.path.join(ROOT_DIR, 'models', 'best534.pt')
UNET_ONNX_PATH = os.path.join(ROOT_DIR, 'models', 'unet_model.onnx')

# Путь для статических данных
RESULT_ABS_PATH = os.path.join(ROOT_DIR, 'static', 'temp', 'result')
DOWN_ABS_PATH = os.path.join(ROOT_DIR, 'static', 'temp', 'download')
UNET_ABS_PATH = os.path.join(ROOT_DIR, 'static', 'temp', 'unet_pred')

# Путь для базы данных
DATA_BASE_PATH = os.path.join(ROOT_DIR, 'instance')  # Путь к базе данных

# Прочие настройки
host = '127.0.0.1'
port = '5000'
debug = None
thread = None

# Пример использования путей
print(f"YOLO модель находится по пути: {YOLO_PATH}")
print(f"Путь к базе данных: {DATA_BASE_PATH}")
print(f"Путь к результатам: {RESULT_ABS_PATH}")
