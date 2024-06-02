import numpy as np
import matplotlib
from rembg import remove
from PIL import Image, ImageOps
import onnxruntime as ort

matplotlib.use('agg')
from sklearn.preprocessing import Binarizer

MODEL_UNET_ONNX = ort.InferenceSession("/home/valeogamer/PycharmProjects/Valgus/App/models/unet_model.onnx")
UNET_PATH = '/home/valeogamer/PycharmProjects/Valgus/UNet/result/'


class Foots:
    def __init__(self):
        self.img_path_orig = None
        self.img_path_unet = None
        self.img_name = None
        self.img_width = None
        self.img_height = None
        self.img_size = None

    def remove_and_black_background(self):
        # Удаление фона с помощью rembg
        input_image = Image.open(self.img_path_orig)
        output_image = remove(input_image)
        self.img_size = output_image.size
        self.img_height, self.img_width = output_image.size

        # Создание черного фона
        black_background = Image.new("RGB", output_image.size, (0, 0, 0))

        # Наложение изображения с прозрачным фоном на черный фон
        black_background.paste(output_image, (0, 0), output_image)

        # Приведение изображения к квадратному виду с помощью отступов
        desired_size = max(self.img_width, self.img_height)
        delta_w = desired_size - self.img_width
        delta_h = desired_size - self.img_height
        padding = (delta_h // 2, delta_w // 2, delta_h - (delta_h // 2), delta_w - (delta_w // 2))
        black_background = ImageOps.expand(black_background, padding, fill='black')

        # Изменение размера изображения до 640x640
        black_background = black_background.resize((640, 640), Image.BICUBIC)

        # Преобразование изображения в массив numpy и стандартизация пиксельных значений
        black_background = np.array(black_background)

        return black_background / 255.

    def pred_unet(self):
        """
        Сегментация изображения
        """
        processed_image = self.remove_and_black_background()
        orig_imgs = [processed_image]

        # # Преобразование изображения в массив numpy и стандартизация пиксельных значений
        img = orig_imgs[0]

        # Расширение размерности для использования модели ONNX
        img = np.expand_dims(img, axis=0)

        # Преобразование типа данных в float32
        img = img.astype(np.float32)

        # Использование модели ONNX для предсказания
        pred = MODEL_UNET_ONNX.run(None, {"input": img})[0]

        pred_mask = Binarizer(threshold=0.5).transform(pred.reshape(-1, 1)).reshape(pred.shape)
        for i in range(len(orig_imgs)):
            combined_image = (orig_imgs[i] * pred_mask[i])
            combined_image = (combined_image * 255.).astype(np.uint8)
            combined_image = Image.fromarray(combined_image)
            ImageOps.fit(combined_image, (640, 640)).save(f'{UNET_PATH}{self.img_name}')

        file_path = f'{UNET_PATH}{self.img_name}'
        self.img_path_unet = file_path
        return file_path


def image_process(img_path=None, file_name=None):
    foots = Foots()
    foots.img_path_orig = img_path
    foots.img_name = file_name
    foots.pred_unet()


if __name__ == '__main__':
    img_path: str = '/home/valeogamer/Загрузки/Unet_BG/00489.png'
    img_name: str = img_path[-9:]
    image_process(img_path, img_name)
