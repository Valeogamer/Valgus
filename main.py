import remove_bg
import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(path_to_image: str):
    """
    Чтение изображения для opencv
    :param path_to_image: путь до изображения
    :return: cv2.imread
    """
    return cv2.imread(path_to_image)


if __name__ == '__main__':
    image_path = 'image_foot_3.png'
    image = read_image(path_to_image=image_path)
    # Преобразование в черно-белое
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Сохраните черно-белое изображение
    cv2.imwrite('b_w.jpg', gray_image)

    # Загрузите черно-белое изображение
    gray_image = cv2.imread('b_w.jpg', cv2.IMREAD_GRAYSCALE)

    # Задайте пороговое значение (например, 128)
    threshold = 128

    # Создайте пустую матрицу для бинарного изображения
    binary_image = gray_image.copy()

    # Примените пороговое значение
    binary_image[binary_image <= threshold] = 0  # Черный
    binary_image[binary_image > threshold] = 255  # Белый

    # Сохраните бинарное изображение
    cv2.imwrite('bin_img.jpg', binary_image)

    # Загрузите бинарное изображение
    binary_image = cv2.imread('bin_img.jpg', cv2.IMREAD_GRAYSCALE)

    # Преобразуйте в NumPy массив для удобства работы
    binary_matrix = np.array(binary_image)

    # Создайте матрицу, где черный цвет представлен 0, а белый цвет 1
    binary_matrix = binary_matrix.astype(np.uint8) // 255

    # Сохраните бинарное изображение в формате PBM
    with open('binary_image.pbm', 'wb') as f:
        f.write(b'P1\n')
        f.write(f'{binary_matrix.shape[1]} {binary_matrix.shape[0]}\n'.encode())
        for row in binary_matrix:
            f.write(' '.join(map(str, row)).encode() + b'\n')

    print()