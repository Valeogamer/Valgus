"""
    Формирование бинарной маски.
"""
import cv2
import os


def binarize_image(image_path, output_directory, threshold=127):
    # Загрузка изображения
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Бинаризация изображения
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Получение имени файла
    filename = os.path.basename(image_path)

    # Получение пути для сохранения бинаризованного изображения
    output_image_path = os.path.join(output_directory, filename)

    # Сохранение результата
    cv2.imwrite(output_image_path, binary_image)


def binarize_images_in_directory(input_directory, output_directory, threshold=127):
    # Создание выходного каталога, если он не существует
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Обработка каждого изображения во входном каталоге
    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_image_path = os.path.join(input_directory, filename)
            binarize_image(input_image_path, output_directory, threshold)


# Пример использования
input_directory = 'C:/Users/Valentin/Desktop/FingerImgBg/'
output_directory = 'C:/Users/Valentin/Desktop/FingerImgMasks/'
threshold = 15
binarize_images_in_directory(input_directory, output_directory, threshold)
