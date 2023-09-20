"""
На данный момент обработка одного фото занимает 1.5 сек
"""
from rembg import remove
from PIL import Image
import time

def remove_my_bg():
    start = time.time()
    input_path = 'images/ (168).jpg'
    output_path = 'iamge_foot_1.png'

    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)
    end = time.time()
    print(end - start)

