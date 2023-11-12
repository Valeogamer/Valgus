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
    remove_bg.remove_my_bg()