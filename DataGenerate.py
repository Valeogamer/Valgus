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

    def rotate(self, image, angle):
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        return rotated_image

    def flip(self, image, flip_type):
        return cv2.flip(image, flip_type)

    def scale(self, image, scale_factor):
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    def translate(self, image, shift_x, shift_y):
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
        return translated_image

    def adjust_brightness(self, image, factor):
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)

    def adjust_contrast(self, image, factor):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.convertScaleAbs(gray_image, alpha=factor, beta=0)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    def add_noise(self, image, mean=0, sigma=25):
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    def crop(self, image, x, y, width, height):
        return image[y:y + height, x:x + width]

    def color_augmentation(self, image, alpha_range=0.5, beta_range=30):
        alpha = 1.0 + np.random.uniform(-alpha_range, alpha_range)
        beta = np.random.uniform(-beta_range, beta_range)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def blur(self, image, kernel_size=(5, 5)):
        return cv2.GaussianBlur(image, kernel_size, 0)

    def gamma_adjustment(self, image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(image, table)

    def save_image(self, image, save_path):
        cv2.imwrite(save_path, image)


class ImageAugmentorPillow:
    def __init__(self):
        pass

    def rotate(self, image, angle):
        return image.rotate(angle)

    def flip(self, image, flip_type):
        if flip_type == 0:  # Отражение по вертикали
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif flip_type == 1:  # Отражение по горизонтали
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            return image

    def scale(self, image, scale_factor):
        new_size = tuple(int(dim * scale_factor) for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def translate(self, image, shift_x, shift_y):
        return image.transform(image.size, Image.Transform.AFFINE, (1, 0, shift_x, 0, 1, shift_y))

    def adjust_brightness(self, image, factor):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def adjust_contrast(self, image, factor):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def add_noise(self, image, mean=0, sigma=25):
        image_array = np.array(image)
        noise = np.random.normal(mean, sigma, image_array.shape)
        noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image_array)

    def crop(self, image, x, y, width, height):
        return image.crop((x, y, x + width, y + height))

    def color_augmentation(self, image, alpha_range=0.5, beta_range=30):
        enhancer = ImageEnhance.Color(image)
        alpha = 1.0 + np.random.uniform(-alpha_range, alpha_range)
        enhanced_image = enhancer.enhance(alpha)
        beta = np.random.uniform(-beta_range, beta_range)
        return ImageEnhance.Brightness(enhanced_image).enhance(1 + beta / 100)

    def blur(self, image):
        return image.filter(ImageFilter.BLUR)

    def gamma_adjustment(self, image, gamma=1.0):
        image_array = np.array(image)
        gamma_corrected_array = np.power(image_array / 255.0, gamma) * 255.0
        return Image.fromarray(gamma_corrected_array.astype(np.uint8))

    def save_image(self, image, save_path):
        image.save(save_path)

class ImageAugmentorTF:
    """
    Слишком сильно зависим от версии
    """
    def __init__(self):
        pass


# Пример использования класса
if __name__ == "__main__":
    start = cv
    if start is cv:
        # Загрузка изображения
        image = cv2.imread("TestImages/Foot.png")

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
        image_pillow = Image.open(image_path)

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
        pass
    elif start is None:
        print("Не выбран кейс!")
    else:
        raise Exception("Ошибка, потерян start")
