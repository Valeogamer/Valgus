"""
 Модуль для автоматического удаления заднего фона.
 Вспомогательный модуль для подготовки датасета.
"""
from rembg import remove
from PIL import Image
import os
import numpy as np

i_path = "C://Users//Valentin//Desktop//ОригинальныеДанныеРассширенный//"
input_path: list[str] = os.listdir(path=i_path)
output_path: str = 'C://Users//Valentin//Desktop//OrigAddMasks//'
background_color = (0, 0, 0)

for path_img in input_path:
    input = Image.open(i_path+path_img)
    output = remove(input)
    background_mask = np.all(input == [0, 0, 0], axis=-1)
    input[background_mask] = background_color
    output.save(output_path + path_img)
