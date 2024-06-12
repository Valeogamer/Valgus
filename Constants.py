# Выбор библиотеки для аугментации данных
CV: str = 'opencv'
PW: str = 'pillow'
# P - Orig
# O - Mask
# Пути до файлов
# Путь до основных каталогов
PATH_DIR = "C://Users//Valentin//Desktop"
# путь до файлов содержащие Оригинальные данные
PATH_DIR_Orig: str = "C://Users//Valentin//Desktop//100Foot//538"  # путь до оригинала
# путь до файлов содержащие Данные масок
PATH_DIR_Mask: str = "C://Users//Valentin//Desktop//100Foot//538Masks"  # путь до масок

# Пути для сохранения аргументированных данных
NEW_PATH_DIR: str = "C://Users//Valentin//Desktop"  # основной путь до хранилища
NEW_PATH_DIR_Orig: str = "C://Users//Valentin//Desktop//100Foot//538Aug"  # путь для сохранения аргументированных данных (оригинала)
NEW_PATH_DIR_Mask: str = "C://Users//Valentin//Desktop//100Foot//538AugMasks"  # путь для сохранения аргументированных данных (масок)

# Новое имя и рассширения для файлов
NAME_Orig: str = '000'
NAME_Mask: str = '000'
EXTENTION_PNG: str = '.png'
EXTENTION_JPG: str = '.jpg'
NAME_Orig_E: str = NAME_Orig + EXTENTION_PNG
NAME_Mask_E: str = NAME_Mask + EXTENTION_PNG
# Размеры для стандартизации
WIDTH: int = 640
HEIGHT: int = 640
