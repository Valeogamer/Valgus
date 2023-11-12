"""
Модуль обработки изображения
Удаление заднего фона
На данный момент обработка одного фото занимает 1.5 сек
"""
from rembg import remove
from PIL import Image
from time import perf_counter
import os
import cv2

def remove_my_bg():
    """
    Удаление заднего фона из изображений
    """
    input_folder = 'dataset_images/'
    output_folder = 'images_remove_bg/'
    masks = 'masks/'

    # Проверяем, существует ли папка для выходных изображений, и создаем ее, если не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Получаем список файлов в папке 'dataset_images/'
    files = os.listdir(input_folder)

    for i, filename in enumerate(files):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f'{filename[:-4]}.png')
        input_image = Image.open(input_path)
        # input_image = input_image.convert("L") # перевод в чб
        output_image = remove(input_image)
        output_image.save(output_path)

# Вызываем функцию для удаления задних фонов
remove_my_bg()
