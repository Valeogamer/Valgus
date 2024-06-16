import os

import cv2
import numpy as np

# Глобальные переменные для хранения координат точек и изображения
points = []
img = None
img_copy = None


# Функция для обработки кликов мыши
def click_event(event, x, y, flags, params):
    global points, img, img_copy
    if event == cv2.EVENT_LBUTTONDOWN:  # Левая кнопка мыши - добавление точки
        points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)  # Уменьшен размер точек
        if len(points) == 3:
            calculate_angle(points)
        cv2.imshow('image', img)
    elif event == cv2.EVENT_RBUTTONDOWN:  # Правая кнопка мыши - удаление последней точки
        if points:
            points.pop()
            img = img_copy.copy()
            for point in points:
                cv2.circle(img, point, 3, (0, 0, 255), -1)  # Уменьшен размер точек
            if len(points) == 3:
                calculate_angle(points)
            else:
                cv2.imshow('image', img)
    cv2.imshow('image', img)


# Функция для вычисления угла между тремя точками
def calculate_angle(pts):
    img_temp = img.copy()
    pt1, pt2, pt3 = pts

    # Векторы
    vec1 = np.array(pt1) - np.array(pt2)
    vec2 = np.array(pt3) - np.array(pt2)

    # Рисуем линии между точками
    cv2.line(img_temp, pt1, pt2, (0, 255, 0), 1)  # Сделана линия потоньше
    cv2.line(img_temp, pt2, pt3, (0, 255, 0), 1)  # Сделана линия потоньше

    # Угол между векторами
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(cos_theta)

    # Преобразуем угол из радиан в градусы
    angle_degrees = np.degrees(angle)

    print(f'Угол между точками: {angle_degrees:.2f} градусов')

    # Вычитаем получившийся угол из 180 градусов
    angle_diff = 180 - angle_degrees

    print(f'Разность с 180 градусами: {angle_diff:.2f} градусов')

    # Вывод угла на изображение
    cv2.putText(img_temp, f'{angle_degrees:.2f} degrees', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_temp, f'Difference from 180: {angle_diff:.2f} degrees', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    cv2.imshow('image', img_temp)


# Загрузка изображения
image_path = 'C:/Users/Valentin/Desktop/Img/'
for img_n in os.listdir(image_path):
    img = cv2.imread(image_path + img_n)

    if img is None:
        print(f"Не удалось загрузить изображение по пути: {image_path}")
    else:
        # Масштабирование изображения для удобства работы
        max_dim = 800
        scale_factor = min(max_dim / img.shape[0], max_dim / img.shape[1])
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        img_copy = img.copy()
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
