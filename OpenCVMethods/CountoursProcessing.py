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
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import splprep, splev
#
# # Загрузка изображения
# image = cv2.imread("0009.png")
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
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Загрузка изображения
image = cv2.imread("0009.png")

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение порогового преобразования для получения бинарного изображения
_, binary_image = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)

# Поиск контуров на бинарном изображении
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contours[0] - нога 1
# contours[1]- нога 2

# Выбор контура с наибольшей площадью (должен быть контур заднего отдела стопы)
largest_contour = max(contours, key=cv2.contourArea)

# Извлечение координат x и y из контура
x = largest_contour[:, 0, 0]
y = largest_contour[:, 0, 1]

# Аппроксимация контура с помощью сглаживания
epsilon = 0.001 * cv2.arcLength(largest_contour, True)
approximated_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

# Извлечение координат x и y из аппроксимированного контура
x_smoothed = approximated_contour[:, 0, 0]
y_smoothed = approximated_contour[:, 0, 1]

# Создание маски из сглаженного контура
mask_smoothed = np.zeros_like(gray_image)
cv2.drawContours(mask_smoothed, [approximated_contour], -1, (255), thickness=cv2.FILLED)

# Наложение маски на исходное изображение
result_image = cv2.bitwise_and(image, image, mask=mask_smoothed)

# Построение кривых и изображений
plt.figure(figsize=(16, 6))

# Исходное изображение с нарисованным контуром
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.plot(x, y, color='red')  # Отображение исходного контура
plt.title('Исходное изображение с контуром')
plt.axis('off')

# Сглаженный контур
plt.subplot(1, 3, 2)
plt.plot(x_smoothed, y_smoothed, color='blue')  # Инвертируем y, чтобы кривая шла вверх
plt.title('Сглаженный контур заднего отдела стопы')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по обеим осям
plt.grid(True)

# Изображение с наложенной маской сглаженного контура
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение с наложенной маской сглаженного контура')
plt.axis('off')

plt.tight_layout()
plt.show()
