#  для запуска и выбора кейсов
CV: str = 'opencv'
PW: str = 'pillow'
TFI: str = 'tfimage'

# Пути до файлов
# ToDo Все пути отладочные временные
# Путь до основных каталогов
PATH_DIR = "TestImages"  # "c://Users//User//Desktop//Разметка Тесты//BFootBalance"
# путь до файлов содержащие данные здоровых
PATH_DIR_P: str = "TestImages//Pronation"  # "c://Users//User//Desktop//Разметка Тесты//BFootBalance//Pronation"
# путь до файлов содержащие данные больных
PATH_DIR_O: str = "TestImages//Overpronation"  # "c://Users//User//Desktop//Разметка Тесты//BFootBalance//Overpronation"

# Новое имя для файлов и расширение
NAME_P: str = 'pronation'
NAME_O: str = 'overpronation'
EXTENTION_PNG: str = '.png'
EXTENTION_JPG: str = '.jpg'
NAME_P_E: str = NAME_P + EXTENTION_PNG
NAME_O_E: str = NAME_O + EXTENTION_PNG
# Размеры для стандартизации
WIDTH: int = 256
HEIGHT: int = 256
