#  для запуска и выбора кейсов
CV: str = 'opencv'
PW: str = 'pillow'
TFI: str = 'tfimage'

# Пути до файлов
# ToDo Все пути отладочные временные
# Путь до основных каталогов
PATH_DIR = "Test"  # "c://Users//User//Desktop//Разметка Тесты//BFootBalance"
# путь до файлов содержащие данные здоровых
PATH_DIR_P: str = "c://Users//User//Desktop//Savefoot//BigFootBackupForMasks//Pronation_1//Pronation" # "c://Users//User//Desktop//Savefoot//BigFootBalance//Overpronation//" # "Test//Pronation"  # "TestImages//Pronation"  # "c://Users//User//Desktop//Разметка Тесты//BFootBalance//Pronation"
# путь до файлов содержащие данные больных
PATH_DIR_O: str = "c://Users//User//Desktop//Savefoot//BigFootBackupForMasks//Overpronation_1//Overpronation " # "c://Users//User//Desktop//Savefoot//BigFootBalance//Pronation//" # "Test//Overpronation"  # "TestImages//Overpronation"  # "c://Users//User//Desktop//Разметка Тесты//BFootBalance//Overpronation"

# Путь каталога для хранения папок
NEW_PATH_DIR: str = "AugBigFoot"
NEW_PATH_DIR_O: str = "AugBigFoot//Overpronation"
NEW_PATH_DIR_P: str = "AugBigFoot//Pronation"

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
