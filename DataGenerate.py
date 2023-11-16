"""
    Аугментация данных.
    Модуль содержит различные классы и их методы для аугментации данных (генерации данных).
    Класс образованный с применением методов OpenCV
    Класс образованный с применением методов TF.keras.preprocessing.image (ImageDataGenerator)
    Классы как минимум должны включать в себя такие методы как:
          - Изменение контрастности и яркости
          - Угол поворота изображения
          - Приближение и отдаление случайных участков
          - Случайное смещение
          - Срез случайной части фото
          - Добавление шума
    Несколько различных классов с целью оценки производительности и изучения различных доступных методов аугментации.

"""
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
import numpy as np
import typing as tp

cv: str = 'opencv'
pw: str = 'pillow'
tfi: str = 'tfimage'
start: tp.Optional[str] = None


class ImageAugmentorCV:
    def __init__(self):
        pass

    def __repr__(self):
        return f'ImageAugmentorOpenCV'

    def rotate(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        Вращение изображения на заданный угол [angle]
        :param image: Изображение
        :param angle: Угол вращения
        :return: изображение, перевернутое на заданный угол
        """
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        return rotated_image

    def flip(self, image: np.ndarray, flip_type: int) -> np.ndarray:
        """
        Отражение (переворот)
        :param image: изображение
        :param flip_type: тип отражения
        0: Отражение вокруг вертикальной оси.
        1: Отражение вокруг горизонтальной оси.
        -1: Отражение вокруг обеих осей.
        :return: Отраженное изображение
        """
        return cv2.flip(image, flip_type)

    def scale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Изменение масштаба изображения
        :param image: Изображение
        :param scale_factor: размер масштаба
        s_f > 1 - увеличение
        s_f < 1 - уменьшение
        :return: Масштабируемое изображение
        """
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    def translate(self, image: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
        """
        Смещение изображения (перенос) на заданные координаты
        :param image: Изображение
        :param shift_x: значение смещения по горизонтали
        :param shift_y: значение смещения по вертикали
        :return: Смещенное изображение
        """
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
        return translated_image

    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Изменение яркости
        :param image: Изображение
        :param factor: регулировка яркости
        (factor > 1) - яркость выше
        (factor < 1) - яркость ниже
        :return: измененное по яркости изображение
        """
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)

    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Изменение контраста изображения
        :param image: Изображение
        :param factor: регулировка контраста
        (factor > 1) - контраст выше
        (factor < 1) - контраст ниже
        :return: измененное по контрасту изображение
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.convertScaleAbs(gray_image, alpha=factor, beta=0)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    def add_noise(self, image: np.ndarray, mean: int = 0, sigma: int = 25) -> np.ndarray:
        """
        Добавляет шумы в изображение
        :param image: Изображение
        :param mean: Среднее значение гауссовского шума
        :param sigma: Стандартное отклонение гауссовского шума
        :return:
        """
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    def crop(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Обрезает изображение до указанных размеров
        :param image: Изображение
        :param x: Координата x верхнего левого угла области для обрезки.
        :param y: Координата y верхнего левого угла области для обрезки.
        :param width: Ширина области для обрезки.
        :param height: Высота области для обрезки.
        :return: Обрезанное изображение
        """
        return image[y:y + height, x:x + width]

    def color_augmentation(self, image: np.ndarray, alpha_range: float = 0.5, beta_range: int = 30) -> np.ndarray:
        """
        Аугментация цвета изображения
        :param image: Изображение.
        :param alpha_range: Диапазон изменения коэффициента масштабирования (alpha).
        :param beta_range: Диапазон изменения смещения (beta).
        :return:
        """
        alpha = 1.0 + np.random.uniform(-alpha_range, alpha_range)
        beta = np.random.uniform(-beta_range, beta_range)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def blur(self, image: np.ndarray, kernel_size: tuple[int, int] = (5, 5)) -> np.ndarray:
        """
        Размытие изображения
        :param image: Изображение
        :param kernel_size: Размер ядра фильтра гаусса. tuple(int, int)
        :return: Размытое изображение
        """
        return cv2.GaussianBlur(image, kernel_size, 0)

    def gamma_adjustment(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Коррекция гаммы изображения
        :param image: Изображение
        :param gamma: Значение гаммы
        (gamma > 1) Значительно темнее
        (gamma < 1) Значительно светлее
        :return:
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(image, table)

    def save_image(self, image: np.ndarray, save_path: str) -> None:
        """
        Сохранение изображения по указанному пути
        :param image: Изображение
        :param save_path: Путь для сохранения
        :return: Сохраняет изображение
        """
        cv2.imwrite(save_path, image)


class ImageAugmentorPillow:
    def __init__(self):
        pass

    def __repr__(self):
        return f'ImageAugmentorPillow'

    def rotate(self, image, angle: int):
        """
        Вращение изображения на заданный угол [angle]
        :param image: Изображение
        :param angle: Угол вращения
        :return: изображение, перевернутое на заданный угол
        """
        return image.rotate(angle)

    def flip(self, image, flip_type: int):
        """
        Отражение (переворот)
        :param image: изображение
        :param flip_type: тип отражения
        0: Отражение вокруг вертикальной оси.
        1: Отражение вокруг горизонтальной оси.
        :return:
        """
        if flip_type == 0:  # Отражение по вертикали
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif flip_type == 1:  # Отражение по горизонтали
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            return image

    def scale(self, image, scale_factor: float):
        """
        Изменение масштаба изображения
        :param image: Изображение
        :param scale_factor: размер масштаба
        s_f > 1 - увеличение
        s_f < 1 - уменьшение
        :return: Масштабируемое изображение
        """
        new_size = tuple(int(dim * scale_factor) for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def translate(self, image, shift_x: int, shift_y: int):
        """
        Смещение изображения (перенос) на заданные координаты
        :param image:  Изображение
        :param shift_x: значение смещения по горизонтали
        :param shift_y: значение смещения по вертикали
        :return: Смещенное изображение
        """
        return image.transform(image.size, Image.Transform.AFFINE, (1, 0, shift_x, 0, 1, shift_y))

    def adjust_brightness(self, image, factor: float):
        """
        Изменение яркости
        :param image: Изображение
        :param factor: регулировка яркости
        (factor > 1) - яркость выше
        (factor < 1) - яркость ниже
        :return: измененное по яркости изображение
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def adjust_contrast(self, image, factor: float):
        """
        Изменение контраста изображения
        :param image: Изображение
        :param factor: регулировка контраста
        (factor > 1) - контраст выше
        (factor < 1) - контраст ниже
        :return: измененное по контрасту изображение
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def add_noise(self, image, mean: int = 0, sigma: int = 25):
        """
        Добавляет шумы в изображение
        :param image: Изображение
        :param mean: Среднее значение гауссовского шума
        :param sigma: Стандартное отклонение гауссовского шума
        :return:
        """
        image_array = np.array(image)
        noise = np.random.normal(mean, sigma, image_array.shape)
        noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image_array)

    def crop(self, image, x: int, y: int, width: int, height: int):
        """
        Обрезает изображение до указанных размеров
        :param image: Изображение
        :param x: Координата x верхнего левого угла области для обрезки.
        :param y: Координата y верхнего левого угла области для обрезки.
        :param width: Ширина области для обрезки.
        :param height: Высота области для обрезки.
        :return: Обрезанное изображение
        """
        return image.crop((x, y, x + width, y + height))

    def color_augmentation(self, image, alpha_range: float = 0.5, beta_range: int = 30):
        """
        Аугментация цвета изображения
        :param image: Изображение.
        :param alpha_range: Диапазон изменения коэффициента масштабирования (alpha).
        :param beta_range: Диапазон изменения смещения (beta).
        :return:
        """
        enhancer = ImageEnhance.Color(image)
        alpha = 1.0 + np.random.uniform(-alpha_range, alpha_range)
        enhanced_image = enhancer.enhance(alpha)
        beta = np.random.uniform(-beta_range, beta_range)
        return ImageEnhance.Brightness(enhanced_image).enhance(1 + beta / 100)

    def blur(self, image):
        """
        Размытие изображения
        :param image: Изображение
        :return: Размытое изображение
        """
        return image.filter(ImageFilter.BLUR)

    def gamma_adjustment(self, image, gamma: float = 1.0):
        """
        Коррекция гаммы изображения
        :param image: Изображение
        :param gamma: Значение гаммы
        (gamma > 1) Значительно темнее
        (gamma < 1) Значительно светлее
        :return:
        """
        image_array = np.array(image)
        gamma_corrected_array = np.power(image_array / 255.0, gamma) * 255.0
        return Image.fromarray(gamma_corrected_array.astype(np.uint8))

    def save_image(self, image, save_path: str):
        """
        Сохранение изображения по указанному пути
        :param image: Изображение
        :param save_path: Путь для сохранения
        :return: Сохраняет изображение
        """
        image.save(save_path)


class ImageAugmentorTF:
    """
    ! Слишком сильно зависим от версии
    """

    def __init__(self):
        pass


# Пример использования класса
if __name__ == "__main__":
    start = pw  # Выбор кейса
    if start is cv:
        # Загрузка изображения
        image: np.ndarray = cv2.imread("TestImages/Foot.png")

        # Создание объекта класса
        augmentor = ImageAugmentorCV()

        # Применение методов аугментации
        rotated_image = augmentor.rotate(image, 30)
        flipped_image = augmentor.flip(image, 1)  # 1 для отражения по вертикали
        scaled_image = augmentor.scale(image, 1.5)
        translated_image = augmentor.translate(image, 50, -30)
        brightened_image = augmentor.adjust_brightness(image, 1.5)
        contrasted_image = augmentor.adjust_contrast(image, 1.5)
        noisy_image = augmentor.add_noise(image)
        cropped_image = augmentor.crop(image, 100, 50, 300, 200)
        color_augmented_image = augmentor.color_augmentation(image)
        blurred_image = augmentor.blur(image)
        gamma_adjusted_image = augmentor.gamma_adjustment(image, gamma=0.8)
        # Сохранение изображений
        augmentor.save_image(rotated_image, "rotated_image.jpg")
        augmentor.save_image(flipped_image, "flipped_image.jpg")
        augmentor.save_image(scaled_image, "scaled_image.jpg")
        augmentor.save_image(translated_image, "translated_image.jpg")
        augmentor.save_image(brightened_image, "brightened_image.jpg")
        augmentor.save_image(contrasted_image, "contrasted_image.jpg")
        augmentor.save_image(noisy_image, "noisy_image.jpg")
        augmentor.save_image(cropped_image, "cropped_image.jpg")
        augmentor.save_image(color_augmented_image, "color_augmented_image.jpg")
        augmentor.save_image(blurred_image, "blurred_image.jpg")
        augmentor.save_image(gamma_adjusted_image, "gamma_adjusted_image.jpg")
    elif start is pw:
        # Pillow
        # Загрузка изображения
        image_path = "TestImages/Foot.png"
        image_pillow = Image.open(image_path)   # :PIL.ImagePlugin(jpeg, png, ...)

        # Создание объекта класса
        augmentor_pillow = ImageAugmentorPillow()

        # Применение методов аугментации
        rotated_image_pillow = augmentor_pillow.rotate(image_pillow, 30)
        flipped_image_pillow = augmentor_pillow.flip(image_pillow, 1)  # 1 для отражения по вертикали
        scaled_image_pillow = augmentor_pillow.scale(image_pillow, 1.5)
        translated_image_pillow = augmentor_pillow.translate(image_pillow, 50, -30)
        brightened_image_pillow = augmentor_pillow.adjust_brightness(image_pillow, 1.5)
        contrasted_image_pillow = augmentor_pillow.adjust_contrast(image_pillow, 1.5)
        noisy_image_pillow = augmentor_pillow.add_noise(image_pillow)
        cropped_image_pillow = augmentor_pillow.crop(image_pillow, 100, 50, 300, 200)
        color_augmented_image_pillow = augmentor_pillow.color_augmentation(image_pillow)
        blurred_image_pillow = augmentor_pillow.blur(image_pillow)
        gamma_adjusted_image_pillow = augmentor_pillow.gamma_adjustment(image_pillow, gamma=0.8)
        # Сохранение изображений
        augmentor_pillow.save_image(rotated_image_pillow, "rotated_image.jpg")
        augmentor_pillow.save_image(flipped_image_pillow, "flipped_image.jpg")
        augmentor_pillow.save_image(scaled_image_pillow, "scaled_image.jpg")
        augmentor_pillow.save_image(translated_image_pillow, "translated_image.jpg")
        augmentor_pillow.save_image(brightened_image_pillow, "brightened_image.jpg")
        augmentor_pillow.save_image(contrasted_image_pillow, "contrasted_image.jpg")
        augmentor_pillow.save_image(noisy_image_pillow, "noisy_image.jpg")
        augmentor_pillow.save_image(cropped_image_pillow, "cropped_image.jpg")
        augmentor_pillow.save_image(color_augmented_image_pillow, "color_augmented_image.jpg")
        augmentor_pillow.save_image(blurred_image_pillow, "blurred_image.jpg")
        augmentor_pillow.save_image(gamma_adjusted_image_pillow, "gamma_adjusted_image.jpg")
    elif start is tfi:
        raise NotImplementedError
    elif start is None:
        print("Не выбран кейс!")
    else:
        raise Exception("Ошибка, потерян start")
