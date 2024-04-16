import cv2
import math
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import Binarizer
from tensorflow.keras.models import load_model

model_yolo = YOLO('best534.pt')
model_unet = load_model('unet_model_other_foot.h5')
IMAGE_SIZE = (640, 640)
PLOTS_DPI = 150


class Foot:
    y_top_params = 50
    y_middle_params = 35
    y_bottom_params = 2
    image = None
    gray = None
    contours = None
    x_contours = y_contours = []
    x_top = y_top = x_middle = y_middle = x_bottom = y_bottom = []

    def __init__(self, type: str):
        self.type: str = type
        self.link = None
        self.x_coords: list = []  # координаты контура
        self.y_coords: list = []  # координаты контура
        self.y_max: int = 0  # макс по Y контура
        self.y_min: int = 0  # мин по Y контура
        self.y_delta: int = 0  # длина по вертикали
        self.y_top_params: int = 60  # % вверхней части
        self.y_middle_params: int = 30  # % средней части
        self.y_bottom_params: int = 10  # % нижней части
        self.y_top: int = 0  # Y вверхней части
        self.y_middle: int = 0  # Y средней части
        self.y_bottom: int = 0  # Y нижней части
        self.y_list_values: list[list[int]] = []  # Возможные варианты значения X на оси на выбранных осях:Y-t;Y-m;Y-b
        self.x_top: int = 0  # X вверхней части
        self.x_middle: int = 0  # X средней части
        self.x_bottom: int = 0  # X нижней части
        # из новых идей
        self.apprx_x_coords: list[int] = []
        self.apprx_y_coords: list[int] = []
        # еще контуры
        self.x_up_l = 0.
        self.y_up_l = 0.
        self.x_down_l = 0.
        self.y_down_l = 0.
        self.x_up_r = 0.
        self.y_up_r = 0.
        self.x_down_r = 0.
        self.y_down_r = 0.

    def __str__(self):
        return f"Foot {self.type}"

    def __repr__(self):
        return f"Foot {self.type}: {id(self)}"

    def ymax_ymin(self):
        """
        Поиск максимального и минимального значения по оси Y.
        """
        self.y_max = max(self.y_coords)
        self.y_min = min(self.y_coords)
        if abs(self.y_max - self.y_min) < 15:
            self.y_min = self.link.y_min
        self.y_delta = int(abs(self.y_max - self.y_min))

    def parameters(self, top: int = 70, middle: int = 25, bottom: int = 1):
        """
        Параметры пропорции.
        """
        self.y_top_params = top
        self.y_bottom_params = bottom
        self.y_middle_params = middle

    def find_y(self):
        """
        Поиск опорной точки по оси Y
        """
        self.ymax_ymin()
        self.y_top = int(self.y_max - (int(self.y_delta * self.y_top_params) / 100))
        self.y_middle = int(self.y_max - (int(self.y_delta * self.y_middle_params) / 100))
        self.y_bottom = int(self.y_max - (int(self.y_delta * self.y_bottom_params) / 100))

    def find_x(self, percent_top: int = 0, mid=False):
        """
        Поиск опорной точки по оси X
        :param: percent_top - процентное соотношение для top
        :param: mid
        """
        # c помощью OpenCV numpy
        indx_b = np.where(self.y_coords == self.y_bottom)
        self.x_bottom = int((self.x_coords[indx_b[0][0]] + self.x_coords[indx_b[0][-1]]) / 2)
        indx_m = np.where(self.y_coords == self.y_middle)
        self.x_middle = int((self.x_coords[indx_m[0][0]] + self.x_coords[indx_m[0][-1]]) / 2)
        # 50%50
        indx_t = np.where(self.y_coords == self.y_top)
        self.x_top = int((self.x_coords[indx_t[0][0]] + self.x_coords[indx_t[0][-1]]) / 2)
        if percent_top != 0:
            # 60%40
            if self.type == 'left':
                delta_l = abs(self.x_coords[indx_t[0][0]] - self.x_coords[indx_t[0][-1]])
                percent = int((percent_top * delta_l) / 100)
                self.x_top = int(max([self.x_coords[indx_t[0][0]], self.x_coords[indx_t[0][-1]]]) - percent)
                # self.x_top = int(((self.x_coords[indx_t[0][0]] + self.x_coords[indx_t[0][-1]]) / 2) + (
                #         (10 * abs(self.x_coords[indx_t[0][0]] - self.x_coords[indx_t[0][-1]])) / 100))  # 60%40
            if self.type == 'right':
                delta_l = abs(self.x_coords[indx_t[0][0]] - self.x_coords[indx_t[0][-1]])
                percent = int((percent_top * delta_l) / 100)
                self.x_top = int(min([self.x_coords[indx_t[0][0]], self.x_coords[indx_t[0][-1]]]) + percent)
                # self.x_top = int(((self.x_coords[indx_t[0][0]] + self.x_coords[indx_t[0][-1]]) / 2) - (
                #         (10 * abs(self.x_coords[indx_t[0][0]] - self.x_coords[indx_t[0][-1]])) / 100))  # 60%40
        if mid:
            self.x_middle = self.x_mid_dot()

    def approx_line(self):
        """
        Вверхняя точка
        """
        # переводим в структуру np.array
        new_x = np.array(self.x_coords)
        new_y = np.array(self.y_coords)
        # извлечение всех координат конутра ниже отметки
        indx = np.where(self.y_middle > new_y)
        new_x = new_x[indx]
        new_y = new_y[indx]
        # извлечение всех координат конутра и выше отметки
        indx = np.where(self.y_min < new_y)
        new_x = new_x[indx]
        new_y = new_y[indx]
        # поиск левого и правого контура
        indx = np.where(self.x_middle > new_x)
        n_x_l = new_x[indx]
        n_y_l = new_y[indx]
        indx = np.where(self.x_middle < new_x)
        n_x_r = new_x[indx]
        n_y_r = new_y[indx]

        # Функции апроксимации y(x)
        degree = 1  # линейная функция апроксимации
        new_Y = np.array([self.y_middle, self.y_min])  # для каких y нужно определить x
        # левая
        coef_l = np.polyfit(n_y_l, n_x_l, degree)
        l_func_yx = np.poly1d(coef_l)
        pred_x_l = l_func_yx(new_Y)
        # правая
        coef_r = np.polyfit(n_y_r, n_x_r, degree)
        r_func_yx = np.poly1d(coef_r)
        pred_x_r = r_func_yx(new_Y)
        ang_l = Foot.angle_between_vectors(pred_x_l[1], self.y_min, pred_x_l[0], self.y_middle,
                                           self.x_middle - self.x_middle / 2, self.y_middle)
        ang_r = Foot.angle_between_vectors(pred_x_r[1], self.y_min, pred_x_r[0], self.y_middle, self.x_middle,
                                           self.y_middle)
        ang_t = Foot.angle_between_vectors(abs((pred_x_r[1] + pred_x_l[1]) / 2), self.y_min, self.x_middle,
                                           self.y_middle, pred_x_l[0], self.y_middle)
        self.x_top = abs((pred_x_r[1] + pred_x_l[1]) / 2)
        self.y_top = self.y_min
        self.x_up_l = pred_x_l[1]
        self.y_up_l = self.y_min
        self.x_down_l = pred_x_l[0]
        self.y_down_l = self.y_middle
        self.x_up_r = pred_x_r[1]
        self.y_up_r = self.y_min
        self.x_down_r = pred_x_r[0]
        self.y_down_r = self.y_middle
        plt.plot(n_x_l, n_y_l)
        plt.plot(n_x_r, n_y_r)
        plt.plot(pred_x_l, new_Y, '-r^')
        plt.plot(pred_x_r, new_Y, '-g^')
        plt.plot([pred_x_l[1], pred_x_l[0], self.x_middle - self.x_middle / 2],
                 [self.y_min, self.y_middle, self.y_middle], '-y*')
        plt.plot([pred_x_r[1], pred_x_r[0], self.x_middle], [self.y_min, self.y_middle, self.y_middle], '-b*')
        plt.plot([abs((pred_x_r[1] + pred_x_l[1]) / 2), self.x_middle, pred_x_l[0]],
                 [self.y_min, self.y_middle, self.y_middle], '-r^')
        plt.gca().invert_yaxis()
        plt.show()
        # print(ang_l)
        # print(ang_r)
        # print(ang_t)
        # между контурами pred_x_l, new_Y и pred_x_r, new_Y построить линию
        return (n_x_l, n_y_l), (n_x_r, n_y_r)

    def x_mid_dot(self):
        left_x_bound, right_x_bound = Foot.find_x_bounds_for_y(self.apprx_x_coords, self.apprx_y_coords, self.y_middle)
        print(left_x_bound, right_x_bound)
        return int((left_x_bound + right_x_bound) / 2)

    @staticmethod
    def angle_between_vectors(x1, y1, x2, y2, x3, y3):
        """
        Определение угла между двумя прямыми.
        """
        # x3 y3 - bot
        # x2 y2 - mid
        # x1 y1 - top
        # Находим координаты векторов AB и BC
        vec_ab = (x2 - x1, y2 - y1)
        vec_bc = (x3 - x2, y3 - y2)

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

    @staticmethod
    def visualization(left, right, apprx_l=False):
        """
        Визуализация
        """
        left: Foot
        right: Foot
        plt.plot(left.x_top, left.y_top, 'r*')
        plt.plot(left.x_middle, left.y_middle, 'g*')
        plt.plot(left.x_bottom, left.y_bottom, 'r*')
        plt.plot([left.x_top, left.x_middle, left.x_bottom], [left.y_top, left.y_middle, left.y_bottom], '-ro')
        plt.plot(right.x_top, right.y_top, 'r*')
        plt.plot(right.x_middle, right.y_middle, 'g*')
        plt.plot(right.x_bottom, right.y_bottom, 'r*')
        plt.plot([right.x_top, right.x_middle, right.x_bottom], [right.y_top, right.y_middle, right.y_bottom],
                 '-co')
        # if apprx_l:
        #     lw = 3
        #     plt.plot([left_foot.x_up_l, left_foot.x_down_l, left_foot.x_middle - left_foot.x_middle / 2],
        #              [left_foot.y_up_l, left_foot.y_down_l, left_foot.y_middle], '-c*', linewidth=lw)
        #     plt.plot([left_foot.x_up_r, left_foot.x_down_r, left_foot.x_middle],
        #              [left_foot.y_up_r, left_foot.y_down_r, left_foot.y_middle], '-b*', linewidth=lw)
        #     plt.plot([abs((left_foot.x_up_l + left_foot.x_up_r) / 2), left_foot.x_middle, left_foot.x_down_l],
        #              [left_foot.y_min, left_foot.y_middle, left_foot.y_middle], '-r^', linewidth=lw)
        #     plt.plot([right_foot.x_up_l, right_foot.x_down_l, right_foot.x_middle - right_foot.x_middle / 4],
        #              [right_foot.y_up_l, right_foot.y_down_l, right_foot.y_middle], '-c*', linewidth=lw)
        #     plt.plot([right_foot.x_up_r, right_foot.x_down_r, right_foot.x_middle],
        #              [right_foot.y_up_r, right_foot.y_down_r, right_foot.y_middle], '-b*', linewidth=lw)
        #     plt.plot([abs((right_foot.x_up_l + right_foot.x_up_r) / 2), right_foot.x_middle, right_foot.x_down_l],
        #              [right_foot.y_min, right_foot.y_middle, right_foot.y_middle], '-r^', linewidth=lw)
        plt.gca().invert_yaxis()
        plt.imshow(Foot.image)
        left_angl = Foot.angle_between_vectors(left_foot.x_top, left_foot.y_top, left_foot.x_middle,
                                               left_foot.y_middle, left_foot.x_bottom, left_foot.y_bottom)
        right_angl = Foot.angle_between_vectors(right_foot.x_top, right_foot.y_top, right_foot.x_middle,
                                                right_foot.y_middle, right_foot.x_bottom, right_foot.y_bottom)
        plt.text(left.x_middle, left.y_middle, f'{left_angl:.04}', fontsize=15, color='blue', ha='right')
        plt.text(right.x_middle, right.y_middle, f'{right_angl:.04}', fontsize=15, color='blue', ha='left')
        # plt.xticks([])
        # plt.yticks([])
        plt.xlabel("Ось X")
        plt.ylabel("Ось Y")
        plt.show()
        # plt.savefig("plot.png")
        # if img_path:
        #     plt.savefig("plot.png")
        # plt.savefig(f'{img_path[-9:-4]}.png', dpi=100)

    @staticmethod
    def image_to_countors(img: str, tresh_begin: int = 25, tresh_end: int = 255):
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
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, tresh_begin, tresh_end, cv2.THRESH_BINARY)
        contours, hierarhy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        return contours, gray, image

    @staticmethod
    def aprox_contours(num_dots=10):
        """
        Аппроксимация контура
        """
        linear = 'linear'
        quadratic = 'quadratic'
        polynomial = 'polynomial'
        cubic = 'cubic'
        kind_choice = linear
        # Получение высоты изображения
        height = Foot.image.shape[0]
        half_height = int(height / 2)
        # Итерирование по каждому контуру
        for contour in Foot.contours:
            # Извлечение координат x и y из контура
            x = contour[:, 0, 0]
            y = contour[:, 0, 1]

            filtered_indices = np.where(y >= half_height)
            x = x[filtered_indices]
            y = y[filtered_indices]

            # Интерполяция данных
            # Делим весь контур на равнные части в интервале от 0 до 1
            t = np.linspace(0, 1, len(y))
            # Собираем функцию аппроксимации
            fx, fy = interp1d(t, x, kind=kind_choice), interp1d(t, y, kind=kind_choice)

            # Определение новых точек контура
            t_new = np.linspace(0, 1, num_dots)  # Увеличьте количество точек для более гладкой аппроксимации
            # наоборот ухудшаю для вырезания торчащих элементов
            x_new, y_new = fx(t_new), fy(t_new)

            if len(right_foot.apprx_x_coords) == 0:
                right_foot.apprx_x_coords = x_new
                right_foot.apprx_y_coords = y_new
            else:
                left_foot.apprx_x_coords = x_new
                left_foot.apprx_y_coords = y_new
            # Наложение исходного контура на изображение
            # cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

            # Наложение аппроксимированного контура на изображение
            approximated_contour = np.array(list(zip(x_new.astype(int), y_new.astype(int))), dtype=np.int32).reshape(
                (-1, 1, 2))
            cv2.polylines(Foot.image, [approximated_contour], isClosed=True, color=(255, 0, 0), thickness=2)

        # Отображение результата
        # plt.imshow(cv2.cvtColor(Foot.image, cv2.COLOR_BGR2RGB))
        # plt.title('Contour Approximation')
        # plt.gca().invert_yaxis()
        # plt.show()

    @staticmethod
    def find_x_bounds_for_y(contour_x, contour_y, target_y, tolerance=50):
        # ToDo Заменить моми методом
        x_bounds = []
        for i in range(len(contour_y)):
            if abs(contour_y[i] - target_y) <= tolerance:
                x_bounds.append(int(contour_x[i]))
        left_x_bound = min(x_bounds)
        right_x_bound = max(x_bounds)
        return left_x_bound, right_x_bound


def yolo_key_point(img_path, l_f, r_f):
    flag = True
    conf_i = 0.10
    while flag:
        results = model_yolo.predict(img_path, conf=conf_i)
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
            l_f.x_middle = l_x[1]
            l_f.y_middle = l_y[1]
            r_f.x_middle = r_x[1]
            r_f.y_middle = r_y[1]
        else:
            r_f.x_middle = l_x[1]
            r_f.y_middle = l_y[1]
            l_f.x_middle = r_x[1]
            l_f.y_middle = r_y[1]


def load_test_image(filepath):
    img = tf.io.read_file(filepath)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img / 255.0


def pred_unet(img_path):
    """
    Сегментация изображения
    """
    orig_imgs = [load_test_image(img_path)]
    pred = model_unet.predict(np.array(orig_imgs))
    pred_mask = Binarizer(threshold=0.5).transform(pred.reshape(-1, 1)).reshape(pred.shape)
    # Создаем директорию для сохранения изображений, если её еще нет
    for i in range(len(orig_imgs)):
        # Сохраняем исходное изображение
        # plt.imshow(pred_imgs[i])
        # plt.axis('off')
        # plt.savefig(f'predictions/test_sample_{i + 1}.jpg', bbox_inches='tight', pad_inches=0)
        # plt.close()
        #
        # # Сохраняем мягкую маску
        # plt.imshow(pred[i], cmap='gray')
        # plt.axis('off')
        # plt.savefig(f'predictions/soft_mask_{i + 1}.jpg', bbox_inches='tight', pad_inches=0)
        # plt.close()
        #
        # # Сохраняем бинарную маску
        # plt.imshow(pred_mask[i], cmap='gray')
        # plt.axis('off')
        # plt.savefig(f'predictions/binary_mask_{i + 1}.jpg', bbox_inches='tight', pad_inches=0)
        # plt.close()

        # Сохраняем маскированное изображение
        plt.imshow(orig_imgs[i] * pred_mask[i])
        plt.axis('off')
        plt.savefig(f'unet_pred/{img_path[-9:]}', bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    import os
    # path = "/home/valeogamer/Загрузки/Unet_BG/"
    # path_imgs = os.listdir(path)
    # for img in path_imgs:
    #     pred_unet(path + img)
    path = 'unet_pred/'
    path_imgs = os.listdir(path)
    # for p_img in path_imgs:
    p_img = '00538.png'
    img_path: str = path + p_img
    # img_path: str = 'C:/Users/Valentin/Desktop/DataTest/00496.png'
    # для обрезания пальцев с помощью апркосимации (средняя точка)
    contour_mid = False
    dots = 5
    mid_x = False

    # для определения вверхней точки апроксимацией (вверхняя точка)
    apprx_line_top = False
    apprx_viz = False
    apply_yolo = True

    # % соотношение вверхней точки
    percent = 45

    # извлекаем контур изображения
    Foot.contours, Foot.gray, Foot.image = Foot.image_to_countors(img_path)
    if len(Foot.contours) > 2:
        sorted_indices = np.argsort([-arr.size for arr in Foot.contours])
        Foot.contours = (Foot.contours[sorted_indices[0]], Foot.contours[sorted_indices[1]])
    # формируем экземпляр левой и правой ноги
    left_foot = Foot("left")
    right_foot = Foot("right")

    # перекрестная ссылка
    left_foot.link = right_foot
    right_foot.link = left_foot

    # извлекаем контур для каждой ноги отдельно
    for contour in Foot.contours:
        if len(right_foot.x_coords) == 0:
            right_foot.x_coords = contour[:, 0, 0]
            right_foot.y_coords = contour[:, 0, 1]
        else:
            left_foot.x_coords = contour[:, 0, 0]
            left_foot.y_coords = contour[:, 0, 1]
        Foot.x_contours.extend(contour[:, 0, 0])
        Foot.y_contours.extend(contour[:, 0, 1])
    if left_foot.x_coords.max() > right_foot.x_coords.max():
        left_foot.x_coords, right_foot.x_coords = right_foot.x_coords, left_foot.x_coords
        left_foot.y_coords, right_foot.y_coords = right_foot.y_coords, left_foot.y_coords

    # % соотношение по высоте Y
    left_foot.parameters(top=90, middle=30, bottom=1)  # процентное сотноошение по высоте с низу вверх
    right_foot.parameters(top=90, middle=30, bottom=1)  # от ступни до голени

    # для обрезания пальцев с помощью апркосимации
    if contour_mid:
        Foot.aprox_contours(num_dots=dots)
        mid_x = True
        apply_yolo = False

    # Поиск необходимых Y
    left_foot.find_y()
    right_foot.find_y()

    # Поиск необходимых X
    left_foot.find_x(percent_top=percent, mid=mid_x)
    right_foot.find_x(percent_top=percent, mid=mid_x)

    # Определение центральной точки с применением YOLO
    if apply_yolo:
        yolo_key_point(img_path, left_foot, right_foot)

    #  Определение вверхней точки с помощью апроксимации двух боковых кривых и нахождения средней кривой между нимим
    if apprx_line_top:
        left_foot.approx_line()
        right_foot.approx_line()
        apprx_viz = True

    # Визуализация
    Foot.visualization(left_foot, right_foot, apprx_l=apprx_viz)

    # Вычисление угла пронации
    print("\033[31m" + str(img_path[-9:]) + "\033[0m")
    print("\033[32m" + f'{left_foot}:' + str(
        Foot.angle_between_vectors(left_foot.x_top, left_foot.y_top, left_foot.x_middle, left_foot.y_middle,
                                   left_foot.x_bottom, left_foot.y_bottom)) + "\033[0m")
    print("\033[32m" + f'{right_foot}:' + str(
        Foot.angle_between_vectors(right_foot.x_top, right_foot.y_top, right_foot.x_middle, right_foot.y_middle,
                                   right_foot.x_bottom, right_foot.y_bottom)) + "\033[0m")
