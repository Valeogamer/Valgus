#  для запуска и выбора кейсов
CV: str = 'opencv'
PW: str = 'pillow'
TFI: str = 'tfimage'

# Пути до файлов
# ToDo Все пути отладочные временные
# Путь до основных каталогов
PATH_DIR = "C://Users//Valentin//Desktop"
# путь до файлов содержащие данные здоровых
PATH_DIR_P: str = "C://Users//Valentin//Desktop//640x640copy//"  # "C://Diplom//Savefoot//BigFootBackupForMasks//Pronation_1//MaskPronationGray"
# путь до файлов содержащие данные больных
PATH_DIR_O: str = "C://Users//Valentin//Desktop//MasksBincopy//"  # "C://Diplom//Savefoot//BigFootBackupForMasks//Overpronation_1//MaskOverpronationGray"

# Путь каталога для хранения папок
NEW_PATH_DIR: str = "C://Users//Valentin//Desktop"
NEW_PATH_DIR_O: str = "C://Users//Valentin//Desktop//NewMasksBinAug"
NEW_PATH_DIR_P: str = "C://Users//Valentin//Desktop//New640x640Aug"

# Новое имя для файлов и расширение
NAME_P: str = '000'
NAME_O: str = '000'
EXTENTION_PNG: str = '.png'
EXTENTION_JPG: str = '.jpg'
NAME_P_E: str = NAME_P + EXTENTION_PNG
NAME_O_E: str = NAME_O + EXTENTION_PNG
# Размеры для стандартизации
WIDTH: int = 640
HEIGHT: int = 640
