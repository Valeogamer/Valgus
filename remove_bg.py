"""
Модуль обработки изображения
Удаление заднего фона
На данный момент обработка одного фото занимает 1.5 сек
"""
from rembg import remove
from PIL import Image
from time import perf_counter

def remove_my_bg():
    """
    Удаление заднего фона
    :return:
    """
    start = perf_counter()
    input_path = 'images/ (168).jpg'
    output_path = 'image_foot_3.png'

    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)
    end = perf_counter()
    print(end - start)
