# Valgus / Определение плоско-вальгусной деформации стопы методами машинного обучения.

___

Данный репозиторий посвящен разработке методов для определения плоско-вальгусной деформации стопы.
А также является результатом ВКР и показателем полученных навыков в течении всего учебного процесса.

## Цель <img align='center' src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Activities/Bullseye.png" alt="Bullseye" width="30" height="30" />

___
Разработать методы для определения по фото плоско-вальгусной деформации стопы.

## Задачи <img align='center' src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Symbols/Triangular%20Flag.png" alt="Triangular Flag" width="30" height="30" />

___
- :white_check_mark: Изучить проблему. Актуальность. Возможности.
- :white_check_mark: Рассмотреть полезные библиотеки.
- :white_check_mark: Разработать варианты решения.
- :white_check_mark: Реализовать алгоритмы.
- :white_check_mark: Реализовать варианты решения.
- :white_check_mark: Тестирование.
- :white_check_mark: Заключение.
- :white_check_mark: Разработать интерфейс GUI, UI/UX.
- :white_check_mark: Готовые варианты решения собрать как веб приложение.
- :white_check_mark: Написать ВКР:
    - :white_check_mark: Введение.
    - :white_check_mark: Глава 1. Теоретическая часть.
    - :white_check_mark: Глава 2. Методология.
    - :white_check_mark: Глава 3. Результаты исследований.
    - :white_check_mark: Глава 4. Веб-приложение.
    - :white_check_mark: Заключение.
    - :white_check_mark: Приложение.
- [ ] : 27/06/2024 Защитить ВКР
- Из мелких задач:
  - :white_check_mark: Реализовать блок-схемы алгоритма.
  - :white_check_mark: Диаграммы (UML)
  - :white_check_mark: Графики (сравнение истинных и предсказанных результатов)
  - :white_check_mark: Иллюстрации
  - :white_check_mark: Листинг кода

## Дневник <img align='center' src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Spiral%20Calendar.png" alt="Spiral Calendar" width="35" height="35" />

___

### - :bookmark: Модуль удаления заднего фона:
  (20/09/2023)
  - Чтение каталога с файлами
  - Удаление фона
  - Запись в новый созданный каталог под теми же именами (наподобие маски под оригинальное изображени)
  - Сохранение результата
### - :bookmark: Формирование датасета из изображения:
  (30/10/2023)
  - Замена имени и расширения
  - Перевод в матричный вид
  - Стандартизация изображения
  - Формирования связки изображение - метка
  - Сохранение результата с помощью numpy и pickle
  - (new) Визуализация (проверка данных соответствует ли метка изображению)
### - :bookmark: Модели НС:
  ### MLP:
  (01/11/2023)
  - проблема в не сбалансированном количестве данных, один класс данных преобладает над другим классом данных
    из-за этого все результаты близятся к результату, того класса у которой больше всего было данных при обучении (
    *__Дисбаланс данных__*).
    В качестве решения данной проблемы, на текущий момент попытаюсь привести данные всех классов к равному количеству.
    А также, чтобы пропорционально увеличить количество данных, произведу *__аугментации данных__*.
    Как вариант можно рассмотреть применение взвешивания классов в функции потерь модели.
  ### CNN: 
  (01/11/2023)
  - Аналогичная проблема
### - :bookmark: Модуль аугментации(генерации) данных
  (16/11/2023)
  - Класс DataGenCV (класс для аугментации данных с применением методов openCV)
  - Класс DataGenTF (класс для аугментации данных с применением методов TF)
  - Класс DataGenPW (класс для аугментации данных с применением методов Pillow)
  - У каждого класса аналогичные методы:
    - Повороты (Rotation): Поворачивание изображения на определенный угол
    - Отражение (Flip): Отражение изображения по горизонтали или вертикали.
    - Масштабирование (Scaling): Изменение размера изображения.
    - Сдвиг (Translation): Смещение изображения по горизонтали или вертикали.
    - Изменение яркости (Brightness adjustment): Изменение яркости изображения.
    - Изменение контраста (Contrast adjustment): Изменение контрастности изображения.
    - Добавление шума (Noise addition): Добавление случайного шума на изображение.
    - Обрезка (Crop): Обрезка изображения до определенного размера.
    - Цветовая аугментация (Color augmentation): Изменение цветовых характеристик изображения.
    - Размытие (Blur): Применение размытия для сглаживания изображения.
    - Комбинированные преобразования (Combined transformations): Совмещение нескольких аугментаций, чтобы создать более разнообразные варианты данных.
    - Изменение гаммы (Gamma adjustment): Изменение гаммы изображения.
    - Эластичные искажения (Elastic distortions): Применение эластичных искажений для искажения изображения.
    - Добавление случайных объектов (Random object insertion): Добавление случайных объектов на изображение.
    - Сегментация (Image segmentation): Разделение изображения на сегменты и их перемещение.


### - :bookmark: Модуль  визуализации (отладочное)
  (16/11/2023)
### - :bookmark: Реализована модель для сегментации на архитектуре U-Net.
  (11/12/2023)
  - Требуется больше данных и повысить точность.
### - :bookmark: Реализованы модели для классфикации.
  (20/12/2023)
  - Как результат, я угадываю лучше чем сетка.
### - :bookmark: Реализованы модели для выделения области интереса и определения ключевых точек.
  (11/01/2024) \
  Дообучена сетка YOLOv8, которая уже умела выделять область интереса и ключевые точки,
  но не на моих данных, были размечены данные в количестве 100шт (фотоизображение) и дообучена сетка. Как результат
  области и точки выделяются, но с большой ошибкой, точность 70%, для более лучшего результата требуется больше данных.

### - :bookmark: Реализовано определение ключевых точек с помощью OpenCV.
  (23/01/2024) \
  Предполагается что уже задний фон удален, находим контуры ног, получаем в виде матрицы.
  Далее находим самую максимальную и минимальную координату, вычисляем длину ноги и применяем пропорцию
  на данный момент сверху 60 - 30 - 10, далее по середине ноги, то есть в каждом замкнутом контуре ставим данные точки для каждой из ног.
  Где 60 - 30 - 10 полученные числа это по оси Y, а середина между двумя замкнутами контурами это X.
  Требуется улучшение. Вычилсять отдельно для каждой ноги. На данный момент высота одна и так же для каждой ноги.

### - :bookmark: Улучшена модель сегментации U-net.
  (07/02/2024) \
  Добавлены новые и аугментированные данные, изменены гиперпараметры. Точность увеличилась до 82%.
  Необходимо еще попробовать поменять гиперпараметры и протестировать.

### - :bookmark: Составлен финальный образ приложения.
  (09/02/2024) \
  *__На входе:__* Фотоизображение. \
  Обработка входящего изображение, определение корректные и адекватные ли данные поступили. (Сетка для отсеивания некорректных данных). \
  Обработка фотоизображения ноги с помощью модели сегментации. \
  Определение области интереса (региона). \
  Определение контуров в данной области, расстановка и определение ключевых точек. \
  Вычисление угла пронации. \
  *__На выходе:__* Изображение с отмеченым углом и соотвествующие числа, также рекомендация и вывод.

### - :bookmark: Удаление заднего, сгементация ног.
  (11/03/2024) \
  Параметры: \
  RANDOM_STATE = 7 \
  IMAGE_SIZE = (640, 640) \
  VAL_SPLIT = 0.1 \
  BATCH_SIZE = 64 \
  SHUFFLE_BUFFER = 400 \
  LR_ALPHA = 0.3 \
  LEARNING_RATE = 1e-4 \
  EPOCHS = 25 \
  PLOTS_DPI = 150 \
  PATIENCE = 3 \
  Количество данных: 15600. \
  Точность: 91%.

### - :bookmark: Определение ключевых точек:
  (16/03/2024) \
  IMAGE_SIZE = (640, 640) \
  EPOCHS = 100 \
  Количество данных: 300 \
  Основано на Ultralytics -> YoLo v8 n - pose. \
  mAp = 98%. \
  precision = 98% \
  recall = 98%

### - :bookmark: Отладка кода.
  (18/03/2024) \
  Исправление ошибок в локальном проекте. \

### - :bookmark: Новый способ определения вверхней точки.
  (12/04/2024) \
  Определяем контуры ног по отдельности, определяем среднюю точку. \
  Для средней точки слева и справа по горизонтали находим координаты контура. \
  Получается есть левая кривая и правая, собираем для каждой функцию апроксимации. \
  Функцию собираем y(x) такого вида для левой и правой части. \
  Определяем x для средней точки и для самой максимально возможной точки y. \
  Соединяем полученные точки, отдельно для каждой из сторон. \

### - :bookmark: Отладка кода.
  (15/04/2024) 
  - Исправлен метод определения левой и правой части координат контура для определенной Y.
  - Исправлены ошибки свяанные с выбором контура.
  - Возможность кооректировки % соотношения как по горизонтали (для вверхней точки), так и по вертикали.
  - Доработана визуализация
  - Исправлены мелкие ошибки

### - :bookmark: Начата реализация веб решения.
  (16/04/2024) 
  - Выбор фреймворка (Flask/Django/FastAPI)
  - Изучение фреймворка
  - Тестовые страницы

### - :bookmark: Реализация макета страниц веб приложения.
  (21/04/2024) 
  - Основы Flask/SqlAlchemy
  - Реализованы макеты страниц Home/Tutorial/About

### - :bookmark: Написана теоретическая часть ВКР.
  (04/05/2024) 
  - Обзор литературы
  - Основы НС, СНС, U-Net, YOLO

### - :bookmark: flask.
  (07/05/2024) 
  - Реализован функционал IO.

### - :bookmark: flask.
  (08/05/2024)
  - Подключение потоковой обработки.

### - :bookmark: flask.
  (09/05/2024) 
  - Подключение модели сегментации U-Net.

### - :bookmark: flask.
  (13/05/2024)
  - Оптимизация программы.
  - Обеспечение потокобезопасности программы.
  - Тестирование потоковой обработки.
  - Исправлена ошибка утечки памяти из-за модуля matplotlib.

### - :bookmark: flask.
  (19/05/2024)
  - Добавлена информация о пользователе (имя, возраст)

### - :bookmark: back.
  (30/05/2024)
  - Теперь точно исправлена ошибка с размерами изображений, при изменении размера изображения теперь пропорции сохраняются

### - :bookmark: Модули.
  (02/06/2024)
  - Все методы отдельно собраны как отдельные модули
  - Сегментация ног (UNet)
  - Пропорциональный метод (Proportional)
  - Метод на основе YOLO (YOLO)
  - Комбинированный метод YOLO + Пропорциональный (Proportional + YOLO)
  - Перевод модели на ONNX
  - Песочница для тестов и идей (Sandbox)

### - :bookmark: back+front.
  (12/06/2024)
  - Реализована страница руководства
  - Доработана логика страницы вывода результата (симптомы/степень)
  - Дополнена страница о нас 
  - Дописан ВКР (требуется проверка)

### - :bookmark: back+front.
  (15/06/2024)
  - Обработка исключений
  - Прототип приложения +- работает
  - В диплом добавлено Приложение А, листинги методов
  - Составлена презентация к защите ВКР
  - Прикреплены таблицы с результатами и соотвествующие изображения

### - :bookmark: back+front.
  (22/06/2024)
  - Создана база данных "DBFoot" (Flask-SQL_Alchemy)
  - Улучшена страница вывода результатов
  - Доработана обработка исключений


## Результат
___
  Реализованы 3 метода определения ключевых анатомических точек.
  Реализован прототип веб-приложение для проведения будущих клинических тестов.

## Полезные ссылки

___

- :pushpin: [МОИ ЗАМЕТКИ ПО ВКР(Notion)](https://valeogamer.notion.site/6b1b24f878ef4167a9469d566dcf8406?pvs=4)
- :pushpin: [МОИ ЗАМЕТКИ ПО Python(Notion)](https://www.notion.so/valeogamer/Python-a38d6b05555f4a329b6b2f30603e1f70?pvs=4)
- :pushpin: [Roboflow (инструмент для разметки данных)](https://docs.roboflow.com/)
- :pushpin: [YOLO](https://www.ultralytics.com/ru)
- :pushpin: [OpenCV](https://docs.opencv.org/3.4/d6/d00/tutorial_py_root.html)
- :pushpin: [Tutor по YOLO](https://www.youtube.com/@Koldim2001)
- :pushpin: [Определение угла пронации стопы 3 точками](https://www.mdpi.com/2076-3417/12/13/6764)
- :pushpin: [Определение угла пронации стопы 3 точками](https://bmcsportsscimedrehabil.biomedcentral.com/articles/10.1186/s13102-022-00457-7)
- :pushpin: [Определение угла пронации стопы 4 точками](https://www.researchgate.net/publication/311246664_Clinical_measures_of_static_foot_posture_do_not_agree)
- :pushpin: [Пронация(видео)](https://youtu.be/7ec8YnKBCt0?si=XBUyKiy460pbOQat)
- :pushpin: [ПВДС](http://vestnik.krsu.edu.kg/archive/15/1139)



<img align='center' src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Alien%20Monster.png" alt="Alien Monster" width="25" height="25" /> @Valeogamer, 2024  <img align='center' src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Alien%20Monster.png" alt="Alien Monster" width="25" height="25" />

