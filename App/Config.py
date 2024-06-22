import os

ROOT_DIR = os.path.abspath("App")
ROOT_DIR = ROOT_DIR.split('\\')
del ROOT_DIR[-1]
ROOT_DIR = '/'.join(ROOT_DIR)
# linux
YOLO_PATH = ROOT_DIR + '/models/best534.pt'
UNET_ONNX_PATH = ROOT_DIR + '/models/unet_model.onnx'
RESULT_PATH = '/static/temp/result/'
RESULT_ABS_PATH = ROOT_DIR + '/static/temp/result/'
DOWN_PATH = '/static/temp/download/'
DOWN_ABS_PATH = ROOT_DIR + '/static/temp/download/'
UNET_PATH = '/static/temp/unet_pred/'
UNET_ABS_PATH = ROOT_DIR + '/static/temp/unet_pred/'

host = '192.168.0.12'
port = '5000'
debug = True
thread = True