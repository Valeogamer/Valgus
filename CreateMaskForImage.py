"""
    Формирование маски для изображений.
    Для НС, которая выделяет объект.
"""
# from PIL import Image, ImageFilter
# from rembg import remove
#
# img = Image.open("BigFootBackupForMasks/images_remove_bg_1/pronation.1.png")
# img_gray = img.convert("L")
# edges = img_gray.filter(ImageFilter.FIND_EDGES)
# edges.show()


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
"""
    Формирование маски для изображений.
    Для НС, которая выделяет объект.
"""
# Путь к изображению
image_path = 'BigFootBackupForMasks/images_remove_bg_1/'

# Загрузка изображения
image = Image.open(image_path)

# Преобразование изображения в оттенки серого
gray_image = image.convert('L')

# Применение пороговой обработки для создания маски
threshold = 50
thresholded_mask = gray_image.point(lambda p: p > threshold and 255)
thresholded_mask.save('new.png')
