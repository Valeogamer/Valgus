import cv2
import matplotlib.pyplot as plt
import math

class Foot:
    y_top_params = 60
    y_middle_params = 25
    y_bottom_params = 4
    image = None
    gray = None
    countours = None
    x_countours = y_countours = []
    x_top = y_top = x_middle = y_middle = x_bottom = y_bottom = []

    def __init__(self, type: str):
        self.type: str = type
        self.link = None
        self.x_coords: list = []  # координаты контура
        self.y_coords: list = []  # координаты контура
        self.y_max: int = 0  # макс по Y контура
        self.y_min: int = 0  # мин по Y контура
        self.y_delta: int = 0  # длина высоты
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

    def __repr__(self):
        return f"Foot {self.type}: {id(self)}"

    def ymax_ymin(self):
        """
        Поиск максимального и минимлаьного значения по оси Y.
        """
        self.y_max = max(self.y_coords)
        self.y_min = min(self.y_coords)
        self.y_delta = int(abs(self.y_max - self.y_min))

    def parameters(self, top: int = 60, middle: int = 30, bottom: int = 10):
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
        self.parameters(top=60, middle=30, bottom=10)
        self.y_bottom = int(self.y_max - int(self.y_delta * Foot.y_top_params) / 100)
        self.y_middle = int(self.y_max - int(self.y_delta * Foot.y_middle_params) / 100)
        self.y_top = int(self.y_max - int(self.y_delta * Foot.y_bottom_params) / 100)
        self.y_list_values.append(Foot.gray[int(self.y_top)])
        self.y_list_values.append(Foot.gray[int(self.y_middle)])
        self.y_list_values.append(Foot.gray[int(self.y_bottom)])

    def find_x(self):
        """
        Поиск опорной точки по оси X
        """
        x_all_coords: list[list[int]] = []
        for y_val in self.y_list_values:
            x_coords: list[int] = []
            cnt_exit = 0
            left_border = True
            right_border = False
            for i in range(1, len(y_val) - 1):
                if left_border:
                    # if y_val[i - 1] < 10 and y_val[i - 1] != 0 and y_val[i - 1] != 1 and y_val[i + 1] > y_val[i]:
                    if y_val[i] > 10:
                        x_coords.append(i)
                        left_border = False
                        right_border = True
                        cnt_exit += 1
                if right_border:
                    # if y_val[i - 1] >= y_val[i] and (y_val[i] == 0 or y_val[i] == 1):
                    if y_val[i] < 10:
                        x_coords.append(i)
                        left_border = True
                        right_border = False
                        cnt_exit += 1
                if cnt_exit >= 4:
                    break
            x_all_coords.append(x_coords)
        if self.type == "left":
            self.x_top = int((x_all_coords[0][0] + x_all_coords[0][1]) / 2)
            self.x_middle = int((x_all_coords[1][0] + x_all_coords[1][1]) / 2)
            self.x_bottom = int((x_all_coords[2][0] + x_all_coords[2][1]) / 2)
        else:
            self.x_top = int((x_all_coords[0][2] + x_all_coords[0][3]) / 2)
            self.x_middle = int((x_all_coords[1][2] + x_all_coords[1][3]) / 2)
            self.x_bottom = int((x_all_coords[2][2] + x_all_coords[2][3]) / 2)

    def angle_between_vectors(self, x1, y1, x2, y2, x3, y3):
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
    def run(left_foot, right_foot):
        """
        Основной стэк вызовов
        """
        left_foot: Foot
        right_foot: Foot
        left_foot.find_y()
        right_foot.find_y()
        left_foot.find_x()
        right_foot.find_x()
        Foot.visualization(left_foot, right_foot)

    @staticmethod
    def visualization(left, right):
        """
        Визуализация
        """
        left: Foot
        right: Foot
        if Foot.x_top:
            # plt.plot(Foot.x_countours, Foot.y_countours)
            plt.plot(Foot.x_top, Foot.y_top, 'r*')
            plt.plot(Foot.x_middle, Foot.y_middle, 'g*')
            plt.plot(Foot.x_bottom, Foot.y_bottom, 'r*')
        else:
            # plt.plot(Foot.x_countours, Foot.y_countours)
            # plt.plot(left.x_top, left.y_top, 'r*')
            # plt.plot(left.x_middle, left.y_middle, 'g*')
            # plt.plot(left.x_bottom, left.y_bottom, 'r*')
            plt.plot([left.x_top, left.x_middle, left.x_bottom], [left.y_top, left.y_middle, left.y_bottom], '-ro')
            # plt.plot(right.x_top, right.y_top, 'r*')
            # plt.plot(right.x_middle, right.y_middle, 'g*')
            # plt.plot(right.x_bottom, right.y_bottom, 'r*')
            plt.plot([right.x_top, right.x_middle, right.x_bottom], [right.y_top, right.y_middle, right.y_bottom], '-co')
            plt.gca().invert_yaxis()
            plt.imshow(Foot.image)
            left_angl = left_foot.angle_between_vectors(left_foot.x_top, left_foot.y_top, left_foot.x_middle,
                                                  left_foot.y_middle, left_foot.x_bottom, left_foot.y_bottom)
            right_angl = right_foot.angle_between_vectors(right_foot.x_top, right_foot.y_top, right_foot.x_middle,
                                                   right_foot.y_middle, right_foot.x_bottom, right_foot.y_bottom)
            plt.text(left.x_middle, left.y_middle, f'{int(left_angl)}', fontsize=15, color='blue', ha='right')
            plt.text(right.x_middle, right.y_middle, f'{int(right_angl)}', fontsize=15, color='blue', ha='left')
            plt.show()


def image_to_countors(img: str, tresh_begin: int = 25, tresh_end: int = 255):
    """
    Определение контура изображения.
    :param img: путь до изображения
    :param tresh_begin: начальный порог
    :param tresh_end: конечный порог
    :return: countours, gray, image
    """
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, tresh_begin, tresh_end, cv2.THRESH_BINARY)
    countours, hierarhy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, countours, -1, (0, 255, 0), 2)

    return countours, gray, image


if __name__ == '__main__':
    img_path: str = '0003.png'
    Foot.countours, Foot.gray, Foot.image = image_to_countors(img_path)
    left_foot = Foot("left")
    right_foot = Foot("right")
    left_foot.link = right_foot
    right_foot.link = left_foot
    for indx in range(len(Foot.countours)):
        for foot in Foot.countours[indx]:
            if indx == 0:
                left_foot.x_coords.append(foot[0][0])
                left_foot.y_coords.append(foot[0][-1])
                Foot.x_countours.append(foot[0][0])
                Foot.y_countours.append(foot[0][-1])
            else:
                right_foot.x_coords.append(foot[0][0])
                right_foot.y_coords.append(foot[0][-1])
                Foot.x_countours.append(foot[0][0])
                Foot.y_countours.append(foot[0][-1])
    Foot.run(left_foot, right_foot)
    print(left_foot.angle_between_vectors(left_foot.x_top, left_foot.y_top, left_foot.x_middle, left_foot.y_middle, left_foot.x_bottom, left_foot.y_bottom))
    print(right_foot.angle_between_vectors(right_foot.x_top, right_foot.y_top, right_foot.x_middle, right_foot.y_middle, right_foot.x_bottom, right_foot.y_bottom))

    # image = cv2.imread('000128.png')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    # countours, hierarhy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # image = cv2.drawContours(image, countours, -1, (0, 255, 0), 2)
    # x, y = [], []
    # for i in countours:
    #     for j in i:
    #         for k in j:
    #             x.append(k[0])
    #             y.append(k[-1])
    # y_min = min(y)  # минимум по координате Y
    # y_max = max(y)  # максимум по координате Y
    # y_delta = max(y) - min(y)  # разница между мин макс
    # # схема 10 - 30 - 60
    # y_1 = int(y_max - y_delta / 10)  # 10%
    # y_2 = int(y_max - y_delta * 1 / 3)  # 30%
    # y_3 = int(y_max - y_delta * 2 / 3)  # 60%
    # # После того как вычилслили оси y_1,2,3, необходимо определить Х, на каждой оси их по 2.
    # arrs = []
    # arrs.append(gray[y_1])
    # arrs.append(gray[y_2])
    # arrs.append(gray[y_3])
    # x_all_coords = []
    # for arr in arrs:
    #     x_coords = []
    #     cnt_exit = 0
    #     left_border = True
    #     right_border = False
    #     for i in range(1, len(arr) - 1):
    #         if left_border:
    #             # if arr[i - 1] < 10 and arr[i - 1] != 0 and arr[i - 1] != 1 and arr[i + 1] > arr[i]:
    #             if arr[i] > 10:
    #                 x_coords.append(i)
    #                 left_border = False
    #                 right_border = True
    #                 cnt_exit += 1
    #         if right_border:
    #             # if arr[i - 1] >= arr[i] and (arr[i] == 0 or arr[i] == 1):
    #             if arr[i] < 10:
    #                 x_coords.append(i)
    #                 left_border = True
    #                 right_border = False
    #                 cnt_exit += 1
    #         if cnt_exit >= 4:
    #             break
    #     x_all_coords.append(x_coords)
    # x_1_l = (x_all_coords[0][0] + x_all_coords[0][1]) / 2
    # x_1_r = (x_all_coords[0][2] + x_all_coords[0][3]) / 2
    #
    # x_2_l = (x_all_coords[1][0] + x_all_coords[1][1]) / 2
    # x_2_r = (x_all_coords[1][2] + x_all_coords[1][3]) / 2
    #
    # x_3_l = (x_all_coords[2][0] + x_all_coords[2][1]) / 2
    # x_3_r = (x_all_coords[2][2] + x_all_coords[2][3]) / 2
    # plt.plot(x, y)
    # plt.plot(x_1_l, y_1, 'r*')
    # plt.plot(x_1_r, y_1, 'r*')
    # plt.plot(x_2_l, y_2, 'g*')
    # plt.plot(x_2_r, y_2, 'g*')
    # plt.plot(x_3_l, y_3, 'y*')
    # plt.plot(x_3_r, y_3, 'y*')
    # plt.gca().invert_yaxis()
    # plt.imshow(image)
    # plt.show()
