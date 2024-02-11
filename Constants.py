#  для запуска и выбора кейсов
CV: str = 'opencv'
PW: str = 'pillow'
TFI: str = 'tfimage'

# Пути до файлов
# ToDo Все пути отладочные временные
# Путь до основных каталогов
PATH_DIR = "C://Diplom//Savefoot//BigFootBackupForMasks"
# путь до файлов содержащие данные здоровых
PATH_DIR_P: str = "C://Diplom//Savefoot//BigFootBackupForMasks//Pronation_1//Pronation"  # "C://Diplom//Savefoot//BigFootBackupForMasks//Pronation_1//MaskPronationGray"
# путь до файлов содержащие данные больных
PATH_DIR_O: str = "C://Diplom//Savefoot//BigFootBackupForMasks//Overpronation_1//Overpronation"  # "C://Diplom//Savefoot//BigFootBackupForMasks//Overpronation_1//MaskOverpronationGray"

# Путь каталога для хранения папок
NEW_PATH_DIR: str = "C://Diplom//OrigImage"
NEW_PATH_DIR_O: str = "C://Diplom//OrigImage//Overpronation"
NEW_PATH_DIR_P: str = "C://Diplom//OrigImage//Pronation"

# Новое имя для файлов и расширение
NAME_P: str = 'pronation'
NAME_O: str = 'overpronation'
EXTENTION_PNG: str = '.png'
EXTENTION_JPG: str = '.jpg'
NAME_P_E: str = NAME_P + EXTENTION_PNG
NAME_O_E: str = NAME_O + EXTENTION_PNG
# Размеры для стандартизации
WIDTH: int = 512
HEIGHT: int = 512
