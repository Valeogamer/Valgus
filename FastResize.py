from PIL import Image
import os

# Путь к каталогу с изображениями
directory = "C://Users//Valentin//Desktop//320x320//"

# Получаем список всех файлов в каталоге
files = os.listdir(directory)

# Проходим через каждый файл
for file in files:
    # Проверяем, что файл - изображение
    if file.endswith(('jpeg', 'jpg', 'png', 'gif')):
        # Открываем изображение
        image_path = os.path.join(directory, file)
        img = Image.open(image_path)

        # Изменяем размер до 640x640
        img_resized = img.resize((320, 320))

        # Сохраняем измененное изображение с тем же именем
        img_resized.save(image_path)