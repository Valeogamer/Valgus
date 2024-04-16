"""
    Модуль для проверки возможности обрубки пальцев с помощью контурной апроксимации.
"""
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Загрузка изображения
# image = cv2.imread("0004.png")
#
# # Преобразование изображения в оттенки серого
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Применение порогового преобразования для получения бинарного изображения
# _, binary_image = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)
#
# # Поиск контуров на бинарном изображении
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# # Выбор контура с наибольшей площадью (должен быть контур заднего отдела стопы)
# largest_contour = max(contours, key=cv2.contourArea)
#
# # Извлечение координат x и y из контура
# x = largest_contour[:, 0, 0]
# y = largest_contour[:, 0, 1]
#
# # Построение кривой
# plt.figure(figsize=(8, 6))
# plt.plot(x, -y)  # Инвертируем y, чтобы кривая шла вверх
# plt.gca().invert_yaxis()
# plt.title('Кривая заднего отдела стопы')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().invert_yaxis()  # Инвертируем ось y, чтобы она шла вверх
# plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по обеим осям
# plt.grid(True)
# plt.show()

#---------------------------------#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import splprep, splev
#
# # Загрузка изображения
# image = cv2.imread("0004.png")
#
# # Преобразование изображения в оттенки серого
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Применение порогового преобразования для получения бинарного изображения
# _, binary_image = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)
#
# # Поиск контуров на бинарном изображении
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# # Выбор контура с наибольшей площадью (должен быть контур заднего отдела стопы)
# largest_contour = max(contours, key=cv2.contourArea)
#
# # Извлечение координат x и y из контура
# x = largest_contour[:, 0, 0]
# y = largest_contour[:, 0, 1]
#
# # Построение исходной кривой
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.plot(x, -y)  # Инвертируем y, чтобы кривая шла вверх
# plt.gca().invert_yaxis()
# plt.title('Исходная кривая заднего отдела стопы')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().invert_yaxis()  # Инвертируем ось y, чтобы она шла вверх
# plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по обеим осям
# plt.grid(True)
#
# # Сглаживание кривой
# xy = np.column_stack((x, y))
# tck, _ = splprep(xy.T, s=0, k=3)
# u_new = np.linspace(0, 1, 1000)
# x_new, y_new = splev(u_new, tck)
#
# # Построение сглаженной кривой
# plt.subplot(1, 2, 2)
# plt.plot(x_new, -y_new)  # Инвертируем y, чтобы кривая шла вверх
# plt.gca().invert_yaxis()
# plt.title('Сглаженная кривая заднего отдела стопы')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().invert_yaxis()  # Инвертируем ось y, чтобы она шла вверх
# plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по обеим осям
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()

# -------------------------------------------- #
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Загрузка изображения
# image = cv2.imread("0004.png")
#
# # Преобразование изображения в оттенки серого
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Применение порогового преобразования для получения бинарного изображения
# _, binary_image = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)
#
# # Поиск контуров на бинарном изображении
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# # Выбор контура с наибольшей площадью (должен быть контур заднего отдела стопы)
# largest_contour = max(contours, key=cv2.contourArea)
#
# # Аппроксимация контура с помощью сглаживания
# epsilon = 0.01 * cv2.arcLength(largest_contour, True)
# approximated_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
#
# # Извлечение координат x и y из аппроксимированного контура
# x_smoothed = approximated_contour[:, 0, 0]
# y_smoothed = approximated_contour[:, 0, 1]
#
# # Построение кривой
# plt.figure(figsize=(8, 6))
# plt.plot(x_smoothed, -y_smoothed)  # Инвертируем y, чтобы кривая шла вверх
# plt.gca().invert_yaxis()
# plt.title('Сглаженная кривая заднего отдела стопы')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().invert_yaxis()  # Инвертируем ось y, чтобы она шла вверх
# plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по обеим осям
# plt.grid(True)
# plt.show()

# ------------------------------------------------ #
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import splprep, splev
#
# # Загрузка изображения
# image = cv2.imread("0004.png")
#
# # Преобразование изображения в оттенки серого
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Применение порогового преобразования для получения бинарного изображения
# _, binary_image = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)
#
# # Поиск контуров на бинарном изображении
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# # Выбор контура с наибольшей площадью (должен быть контур заднего отдела стопы)
# largest_contour = max(contours, key=cv2.contourArea)
#
# # Извлечение координат x и y из контура
# x = largest_contour[:, 0, 0]
# y = largest_contour[:, 0, 1]
#
# # Сглаживание кривой
# xy = np.column_stack((x, y))
# tck, _ = splprep(xy.T, s=0, k=3)
# u_new = np.linspace(0, 1, 1000)
# x_new, y_new = splev(u_new, tck)
#
# # Построение исходной и сглаженной кривых
# plt.figure(figsize=(12, 6))
#
# # Исходная кривая
# plt.subplot(1, 2, 1)
# plt.plot(x, -y, label='Исходная кривая')  # Инвертируем y, чтобы кривая шла вверх
# plt.gca().invert_yaxis()
# plt.title('Задний отдел стопы')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().invert_yaxis()  # Инвертируем ось y, чтобы она шла вверх
# plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по обеим осям
# plt.grid(True)
# plt.legend()
#
# # Сглаженная кривая
# plt.subplot(1, 2, 2)
# plt.plot(x_new, -y_new, label='Сглаженная кривая')  # Инвертируем y, чтобы кривая шла вверх
# plt.gca().invert_yaxis()
# plt.title('Сглаженный задний отдел стопы')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().invert_yaxis()  # Инвертируем ось y, чтобы она шла вверх
# plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по обеим осям
# plt.grid(True)
# plt.legend()
#
# plt.tight_layout()
# plt.show()

# ----------------------- Рабочая тема#
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')  # или 'QtAgg', или 'WXAgg', в зависимости от вашего предпочтения
#
# # Загрузка изображения
# image = cv2.imread("/home/valeogamer/Загрузки/DataTest/00488.png")
#
# # Преобразование изображения в оттенки серого
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Применение порогового преобразования для получения бинарного изображения
# _, binary_image = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)
#
# # Поиск контуров на бинарном изображении
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# # Выбор контура с наибольшей площадью (должен быть контур заднего отдела стопы)
# largest_contour = max(contours, key=cv2.contourArea)
#
# # Извлечение координат x и y из контура
# x = largest_contour[:, 0, 0]
# y = largest_contour[:, 0, 1]
#
# # Аппроксимация контура с помощью сглаживания
# epsilon = 0.009 * cv2.arcLength(largest_contour, True)
# approximated_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
#
# # Извлечение координат x и y из аппроксимированного контура
# x_smoothed = approximated_contour[:, 0, 0]
# y_smoothed = approximated_contour[:, 0, 1]
#
# # Построение кривых
# plt.figure(figsize=(12, 6))
#
# # Исходный контур
# plt.subplot(1, 2, 1)
# plt.plot(x, -y)  # Инвертируем y, чтобы кривая шла вверх
# plt.title('Исходный контур заднего отдела стопы')
# plt.xlabel('X')
# plt.ylabel('Y')
# # plt.gca().invert_yaxis()  # Инвертируем ось y, чтобы она шла вверх
# plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по обеим осям
# plt.grid(True)
#
# # Сглаженный контур
# plt.subplot(1, 2, 2)
# plt.plot(x_smoothed, -y_smoothed)  # Инвертируем y, чтобы кривая шла вверх
# plt.title('Сглаженный контур заднего отдела стопы')
# plt.xlabel('X')
# plt.ylabel('Y')
# # plt.gca().invert_yaxis()  # Инвертируем ось y, чтобы она шла вверх
# plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по обеим осям
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()

# ----------- #
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')  # или 'QtAgg', или 'WXAgg', в зависимости от вашего предпочтения
#
# # Загрузка изображения
# image = cv2.imread("/home/valeogamer/Загрузки/DataTest/00488.png")
#
# # Преобразование изображения в оттенки серого
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Применение порогового преобразования для получения бинарного изображения
# _, binary_image = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)
#
# # Поиск контуров на бинарном изображении
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# # contours[0] - нога 1
# # contours[1]- нога 2
#
# # Выбор контура с наибольшей площадью (должен быть контур заднего отдела стопы)
# largest_contour = max(contours, key=cv2.contourArea)
#
# # Извлечение координат x и y из контура
# x = largest_contour[:, 0, 0]
# y = largest_contour[:, 0, 1]
#
# # Аппроксимация контура с помощью сглаживания
# epsilon = 0.001 * cv2.arcLength(largest_contour, True)
# approximated_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
#
# # Извлечение координат x и y из аппроксимированного контура
# x_smoothed = approximated_contour[:, 0, 0]
# y_smoothed = approximated_contour[:, 0, 1]
#
# # Создание маски из сглаженного контура
# mask_smoothed = np.zeros_like(gray_image)
# cv2.drawContours(mask_smoothed, [approximated_contour], -1, (255), thickness=cv2.FILLED)
#
# # Наложение маски на исходное изображение
# result_image = cv2.bitwise_and(image, image, mask=mask_smoothed)
#
# # Построение кривых и изображений
# plt.figure(figsize=(16, 6))
#
# # Исходное изображение с нарисованным контуром
# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.plot(x, y, color='red')  # Отображение исходного контура
# plt.title('Исходное изображение с контуром')
# plt.axis('off')
#
# # Сглаженный контур
# plt.subplot(1, 3, 2)
# plt.plot(x_smoothed, y_smoothed, color='blue')  # Инвертируем y, чтобы кривая шла вверх
# plt.title('Сглаженный контур заднего отдела стопы')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по обеим осям
# plt.grid(True)
#
# # Изображение с наложенной маской сглаженного контура
# plt.subplot(1, 3, 3)
# plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
# plt.title('Исходное изображение с наложенной маской сглаженного контура')
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import splprep, splev
#
# # Загрузка изображения
# image = cv2.imread("/home/valeogamer/Загрузки/DataTest/00488.png")
#
# # Преобразование изображения в оттенки серого
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Применение порогового преобразования для получения бинарного изображения
# _, binary_image = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)
#
# # Поиск контуров на бинарном изображении
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# # Выбор контура с наибольшей площадью (должен быть контур заднего отдела стопы)
# largest_contour = max(contours, key=cv2.contourArea)
#
# # Извлечение координат x и y из контура
# x = largest_contour[:, 0, 0]
# y = largest_contour[:, 0, 1]
#
# # Аппроксимация контура с помощью сплайнов Безье с измененными параметрами
# tck, u = splprep([x, y], s=0, per=True)
#
# # Определение новых точек контура
# u_new = np.linspace(u.min(), u.max(), 15)
# x_new, y_new = splev(u_new, tck)
#
# # Построение исходного контура и аппроксимированного контура
# plt.figure(figsize=(8, 6))
# plt.plot(x, -y, label='Исходный контур')  # Инвертируем y, чтобы кривая шла вверх
# plt.plot(x_new, -y_new, label='Аппроксимированный контур')  # Инвертируем y, чтобы кривая шла вверх
# plt.title('Исходный и аппроксимированный контуры заднего отдела стопы')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по обеим осям
# plt.legend()
# plt.grid(True)
# plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
#
# # Загрузка изображения
# image_path = "/home/valeogamer/Загрузки/DataTest/00490.png"
# image = cv2.imread(image_path)
#
# # Преобразование изображения в оттенки серого
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Применение порогового преобразования для получения бинарного изображения
# _, binary_image = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)
#
# # Поиск контуров на бинарном изображении
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# # Выбор контура с наибольшей площадью (должен быть контур заднего отдела стопы)
# largest_contour = max(contours, key=cv2.contourArea)
#
# # Извлечение координат x и y из контура
# x = largest_contour[:, 0, 0]
# y = largest_contour[:, 0, 1]
#
# # Интерполяция данных
# t = np.linspace(0, 1, len(x))
# fx, fy = interp1d(t, x, kind='cubic'), interp1d(t, y, kind='cubic')
#
# # Определение новых точек контура
# t_new = np.linspace(0, 1, 10)
# x_new, y_new = fx(t_new), fy(t_new)
#
# # Наложение исходного контура на изображение
# cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
#
# # Наложение аппроксимированного контура на изображение
# approximated_contour = np.array(list(zip(x_new.astype(int), y_new.astype(int))), dtype=np.int32)
# cv2.polylines(image, [approximated_contour], isClosed=True, color=(255, 0, 0), thickness=2)
#
# # Отображение результата
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Your Title Here')
# plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
def aprox_countours(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение порогового преобразования для получения бинарного изображения
    _, binary_image = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)

    # Поиск контуров на бинарном изображении
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    left_x = None
    left_y = None
    right_x = None
    right_y = None
    # Итерирование по каждому контуру
    for contour in contours:
        # Извлечение координат x и y из контура
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]

        # Интерполяция данных
        t = np.linspace(0, 1, len(x))
        fx, fy = interp1d(t, x, kind='cubic'), interp1d(t, y, kind='cubic')

        # Определение новых точек контура
        t_new = np.linspace(0, 1, 10)  # Увеличьте количество точек для более гладкой аппроксимации
        x_new, y_new = fx(t_new), fy(t_new)

        if right_x is None:
            right_x = x_new
            right_y = y_new
        else:
            left_x = x_new
            left_y = y_new
        # Наложение исходного контура на изображение
        # cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # Наложение аппроксимированного контура на изображение
        approximated_contour = np.array(list(zip(x_new.astype(int), y_new.astype(int))), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [approximated_contour], isClosed=True, color=(255, 0, 0), thickness=2)

    # Отображение результата
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Contour Approximation')
    # plt.gca().invert_yaxis()
    plt.show()
    return left_x, left_y, right_x, right_y

def find_x_bounds_for_y(contour_x, contour_y, target_y, tolerance=20):
    x_bounds = []

    for i in range(len(contour_y)):
        if abs(contour_y[i] - target_y) <= tolerance:
            x_bounds.append(contour_x[i])

    left_x_bound = min(x_bounds)
    right_x_bound = max(x_bounds)

    return left_x_bound, right_x_bound

if __name__ == '__main__':
    image_path = "/home/valeogamer/Загрузки/DataTest/00490.png"
    left_x, left_y, right_x, right_y = aprox_countours(image_path)
    # Пример использования для левого контура и высоты y_top
    y_top_l = 150  # Здесь нужно использовать конкретное значение y
    left_x_left_bound, left_x_right_bound = find_x_bounds_for_y(left_x, left_y, y_top_l)
    print("Границы по оси X для левого контура и высоты y_top:", left_x_left_bound, left_x_right_bound)

    # Аналогично для правого контура
    y_top_r = 150
    right_x_left_bound, right_x_right_bound = find_x_bounds_for_y(right_x, right_y, y_top_r)
    print("Границы по оси X для правого контура и высоты y_top:", right_x_left_bound, right_x_right_bound)

