import cv2
import math
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
from rembg import remove
from PIL import Image, ImageOps
import onnxruntime as ort
import Constants as const

matplotlib.use('agg')
from sklearn.preprocessing import Binarizer
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

MODEL_UNET_ONNX = ort.InferenceSession(const.MODEL_UNET_ONNX_W)
RESULT_PATH = const.RESULT_PATH_W
DOWN_PATH = const.DOWN_PATH_W
UNET_PATH = const.UNET_PATH_W
MODEL_YOLO = YOLO(const.MODEL_YOLO_W)


class Foots:
    def __init__(self):
        self.img_path_orig = None
        self.img_path_unet = None
        self.img_path_result = None
        self.img_name = None
        self.img_width = None
        self.img_height = None
        self.img_size = None
        self.left_foot = Foot("left")
        self.right_foot = Foot("right")
        self.image = None
        self.gray = None
        self.contours = None
        self.x_contours = []
        self.y_contours = []

    def image_to_countors(self, tresh_begin: int = 25, tresh_end: int = 255):
        """
        Определение контура изображения.
        :param img: путь до изображения
        :param tresh_begin: начальный порог
        :param tresh_end: конечный порог
        :return: contours, gray, image
        retr_tree - внутренние контуры тоже ищет
        retr_external - только наружные контуры
        CHAIN_APPROX_NONE - чтобы все точки контура хранить
        CHAIN_APPROX_SIMPLE - линейная апроксимация контуров
        CHAIN_APPROX_TC89_L1 CHAIN_APPROX_TC89_KCOS - кубические
        """
        image = cv2.imread(self.img_path_unet)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, tresh_begin, tresh_end, cv2.THRESH_BINARY)
        contours, hierarhy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image = cv2.drawContours(image, contours, -1, (0, 0, 0), 2)
        self.contours = contours
        self.gray = gray
        self.image = image

    def visualization(self):
        """
        Визуализация
        """
        plt.clf()
        # fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
        ax.clear()
        ax.plot(self.left_foot.x_top, self.left_foot.y_top, 'r*')
        ax.plot(self.left_foot.x_middle, self.left_foot.y_middle, 'g*')
        ax.plot(self.left_foot.x_bottom, self.left_foot.y_bottom, 'r*')
        ax.plot([self.left_foot.x_top, self.left_foot.x_middle, self.left_foot.x_bottom],
                [self.left_foot.y_top, self.left_foot.y_middle, self.left_foot.y_bottom], '-ro')
        ax.plot(self.right_foot.x_top, self.right_foot.y_top, 'r*')
        ax.plot(self.right_foot.x_middle, self.right_foot.y_middle, 'g*')
        ax.plot(self.right_foot.x_bottom, self.right_foot.y_bottom, 'r*')
        ax.plot([self.right_foot.x_top, self.right_foot.x_middle, self.right_foot.x_bottom],
                [self.right_foot.y_top, self.right_foot.y_middle, self.right_foot.y_bottom],
                '-ro')
        ax.invert_yaxis()
        # Преобразование всех черных пикселей в белые
        image_copy = self.image.copy()
        black_pixels = (image_copy[:, :, 0] == 0) & (image_copy[:, :, 1] == 0) & (image_copy[:, :, 2] == 0)
        image_copy[black_pixels] = [255, 255, 255]
        ax.imshow(image_copy)
        # ax.imshow(self.image)
        left_angl = self.angle_between_vectors(self.left_foot)
        right_angl = self.angle_between_vectors(self.right_foot)
        self.left_foot.angle = int(left_angl)
        self.right_foot.angle = int(right_angl)
        ax.text(self.left_foot.x_middle, self.left_foot.y_middle, f'{left_angl:.04}', fontsize=15, color='blue',
                ha='right')
        ax.text(self.right_foot.x_middle, self.right_foot.y_middle, f'{right_angl:.04}', fontsize=15, color='blue',
                ha='left')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Убираем поля
        plt.savefig(f'{RESULT_PATH}{self.img_name}', bbox_inches='tight',
                    pad_inches=0)
        self.img_path_result = RESULT_PATH + self.img_name

    def angle_between_vectors(self, link):
        """
        Определение угла между двумя прямыми.
        """
        link: Foot
        # Находим координаты векторов AB и BC
        vec_ab = (link.x_middle - link.x_top, link.y_middle - link.y_top)
        vec_bc = (link.x_bottom - link.x_middle, link.y_bottom - link.y_middle)

        # Вычисляем скалярное произведение векторов AB и BC
        dot_product = vec_ab[0] * vec_bc[0] + vec_ab[1] * vec_bc[1]

        # Вычисляем длины векторов AB и BC
        length_ab = math.sqrt((vec_ab[0] ** 2) + (vec_ab[1] ** 2))
        length_bc = math.sqrt((vec_bc[0] ** 2) + (vec_bc[1] ** 2))

        # Вычисляем угол между векторами в радианах
        angle_rad = math.acos(dot_product / (length_ab * length_bc))

        # Преобразуем радианы в градусы
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def yolo_key_point(self, full=False):
        results = None
        flag = True
        conf_i = 0.10
        while flag:
            results = MODEL_YOLO.predict(self.img_path_unet, conf=conf_i, imgsz=(640, 640))
            check_l = []
            for r in results:
                check_l.append(r.keypoints.xy.tolist())
            if len(check_l[0]) == 2:
                # ToDo чтобы не было такого чтобы bb наложены друг на друга
                flag = False  # Ну или можно break
            else:
                conf_i += 0.01
            if conf_i > 0.60:
                raise NotImplemented

        for r in results:
            keypoints_tensor = r.keypoints.xy
            keypoints_list = keypoints_tensor.tolist()

            # Получаем координаты ключевых точек для левой ноги
            left_xy = keypoints_list[0][:3]
            l_x, l_y = [], []
            for xy in left_xy:
                l_x.append(xy[0])
                l_y.append(xy[1])
            # Получаем координаты ключевых точек для правой ноги
            right_xy = keypoints_list[1][:3]
            r_x, r_y = [], []
            for xy in right_xy:
                r_x.append(xy[0])
                r_y.append(xy[1])
            if l_x[1] < r_x[1]:
                if full:
                    self.left_foot.x_top = l_x[0]
                    self.left_foot.y_top = l_y[0]
                    self.left_foot.x_bottom = l_x[2]
                    self.left_foot.y_bottom = l_y[2]
                    self.right_foot.x_top = r_x[0]
                    self.right_foot.y_top = r_y[0]
                    self.right_foot.x_bottom = r_x[2]
                    self.right_foot.y_bottom = r_y[2]
                self.left_foot.x_middle = l_x[1]
                self.left_foot.y_middle = l_y[1]
                self.right_foot.x_middle = r_x[1]
                self.right_foot.y_middle = r_y[1]
            else:
                if full:
                    self.left_foot.x_top = r_x[0]
                    self.left_foot.y_top = r_y[0]
                    self.left_foot.x_bottom = r_x[2]
                    self.left_foot.y_bottom = r_y[2]
                    self.right_foot.x_top = l_x[0]
                    self.right_foot.y_top = l_y[0]
                    self.right_foot.x_bottom = l_x[2]
                    self.right_foot.y_bottom = l_y[2]
                self.right_foot.x_middle = l_x[1]
                self.right_foot.y_middle = l_y[1]
                self.left_foot.x_middle = r_x[1]
                self.left_foot.y_middle = r_y[1]

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


class Foot:

    def __init__(self, type: str):
        self.type: str = type
        self.angle = None  # вычисленный угол
        self.y_max: int = 0  # макс по Y контура
        self.y_min: int = 0  # мин по Y контура
        self.y_delta: int = 0  # длина по вертикали
        self.y_top: int = 0  # Y вверхней части
        self.y_middle: int = 0  # Y средней части
        self.y_bottom: int = 0  # Y нижней части
        self.x_top: int = 0  # X вверхней части
        self.x_middle: int = 0  # X средней части
        self.x_bottom: int = 0  # X нижней части

    def __str__(self):
        return f"Foot {self.type}"

    def __repr__(self):
        return f"Foot {self.type}: {id(self)}"


def image_process(img_path=None, file_name=None):
    foots = Foots()
    foots.img_path_orig = img_path
    foots.img_name = file_name
    foots.pred_unet()
    foots.image_to_countors()
    foots.yolo_key_point(full=True)

    # Визуализация
    foots.visualization()
    #
    print("\033[31m" + str(img_path[-9:]) + "\033[0m")
    print("\033[32m" + f'{foots.left_foot}:' + str(
        foots.angle_between_vectors(foots.left_foot)) + "\033[0m")
    print("\033[32m" + f'{foots.right_foot}:' + str(
        foots.angle_between_vectors(foots.right_foot)) + "\033[0m")
    return foots.left_foot.angle, foots.right_foot.angle


if __name__ == '__main__':
    image_process(const.img_path, const.img_name)
