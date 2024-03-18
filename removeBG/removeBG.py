"""
 Модуль для автоматического удаления заднего фона.
 Вспомогательный модуль для подготовки датасета.
"""
from rembg import remove
from PIL import Image
import os

i_path = "C://Users//Valentin//Desktop//ОригинальныеДанныеРассширенный//"
input_path: list[str] = os.listdir(path=i_path)
output_path: str = 'C://Users//Valentin//Desktop//OrigAddMasks//'

for path_img in input_path:
    input = Image.open(i_path+path_img)
    output = remove(input)
    output.save(output_path + path_img)


