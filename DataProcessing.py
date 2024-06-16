from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import typing as tp
import numpy as np
import pickle
import cv2
import os


class FileManager:
    """
    Менеджер файлов
    """

    def __init__(self, path_dir, path_dir_orig, path_dir_mask, new_n_orig, new_n_mask, extention, new_path_dir,
                 new_path_dir_mask,
                 new_path_dir_orig):
        # ToDo некоторые атрибуты должны быть protected
        self.path_dir: tp.Optional[str] = path_dir  # Constants.PATH_DIR
        self.path_dir_orig: tp.Optional[str] = path_dir_orig  # Constants.PATH_DIR_Orig
        self.path_dir_mask: tp.Optional[str] = path_dir_mask  # Constants.PATH_DIR_Mask
        self.new_name_orig: tp.Optional[str] = new_n_orig  # Constants.NAME_Orig
        self.new_name_mask: tp.Optional[str] = new_n_mask  # Constants.NAME_Mask
        self.extention: tp.Optional[str] = extention  # Constants.EXTENTION_PNG
        self.new_path_dir: tp.Optional[str] = new_path_dir
        self.new_path_dir_mask: tp.Optional[str] = new_path_dir_mask
        self.new_path_dir_orig: tp.Optional[str] = new_path_dir_orig
        self.path_dir_list_orig: tp.Optional[list[str]] = None
        self.path_dir_list_mask: tp.Optional[list[str]] = None
        self.len_data_orig: tp.Optional[int] = None
        self.len_data_mask: tp.Optional[int] = None
        self.result_path_dir_orig: tp.Optional[int] = None
        self.result_path_dir_mask: tp.Optional[int] = None
        self.len_name: int = 0
        if not os.path.exists(new_path_dir):
            os.makedirs(new_path_dir)
            if not os.path.exists(new_path_dir_mask):
                os.makedirs(new_path_dir_mask)
            if not os.path.exists(new_path_dir_orig):
                os.makedirs(new_path_dir_orig)
        if self.path_dir_orig and self.path_dir_mask:
            self.create_path_dir_list()
        if self.path_dir_list_orig != None:
            self.len_data_orig = len(self.path_dir_list_orig)
            self.len_data_mask = len(self.path_dir_list_mask)
            # ToDo добавить проверку на дисбаланс данных (сейчас делается в ручную)

    def __repr__(self):
        return f"FileManager {id(self)}"

    def add_new_dir(self):
        # ToDo Метод позволяющий добавлять новый каталог
        pass

    def del_dir(self):
        # ToDO Метод для удаления каталога
        pass

    def rename_rextention(self, path_dir: str, new_name: str, extention: str):
        """
        Переименование файлов и изменение расширения
        :param path_dir: Путь до каталога с изображениями
        :param new_name: Новое имя
        :param extention: Новое расширение
        :return:
        """
        images_path = [path_dir + '//' + i for i in os.listdir(path_dir)]
        for i in range(1, len(images_path) + 1):
            os.rename(images_path[i - 1], f'{path_dir + "//" + "new_" + new_name}.{i}{extention}')

    def create_path_dir_list(self):
        """
        Список из путей до файлов
        Заполняет атрибуты экземпляра.
        Атрибут содержит путь (ссылку до каждого файла)
        :return: Заполнят атрибуты: path_dir_list_orig; path_dir_list_mask;
        """
        self.path_dir_list_orig = [self.path_dir_orig + '//' + i for i in os.listdir(self.path_dir_orig)]
        self.path_dir_list_mask = [self.path_dir_mask + '//' + i for i in
                                   os.listdir(self.path_dir_mask)]

    def update_result_data(self):
        """
        Количество полученных данных после обработки.
        :return:
        """
        self.result_path_dir_orig = [self.new_path_dir_orig + '//' + i for i in os.listdir(self.new_path_dir_orig)]
        self.result_path_dir_mask = [self.new_path_dir_mask + '//' + i for i in os.listdir(self.new_path_dir_mask)]


class CreateDatasetImages:
    """
    Формирование датасета из изображения
    """

    def __init__(self):
        pass

    def __repr__(self):
        return f'CreateDatasetImages {id(self)}'

    def convert_images_to_array(self, images_path: list[str]) -> np.array:
        """
        Представляет изображение в виде матрицы
        :param images_path: список с путями до изображений
        :return: список содержащий матрицы
        """
        images_data = []
        for image_path in images_path:
            image = Image.open(image_path)
            image = np.array(image)
            images_data.append(image)
        return images_data

    def resized_images(self, width: int, height: int, images_data: list):
        """
        Изменение размера изображения
        :param width: желаемая ширина изображения
        :param height: желаемая ширина изображения
        :param images_data: список с изображениями
        :return: список изображений с измененными размерами
        """
        new_size = (width, height)  # Задайте желаемый размер

        resized_images = []
        for image in images_data:
            image = Image.fromarray(image).convert("RGB")
            image = image.resize(new_size)
            resized_images.append(np.array(image))
        return resized_images

    def save(self, data: np.array, name: str, flag: int = None):
        """
        Сохранение готового данных
        :param data: данные
        :param name: имя данных
        :param flag: формат сохранения: 1 -> pickle(pkl); 2 - numpy(npy)
        :return:
        """
        if flag == 1:
            with open(f'{name}.pkl', 'wb') as data_file:
                pickle.dump(data, data_file, protocol=pickle.HIGHEST_PROTOCOL)
        elif flag == 2:
            np.save(f'{name}.npy', data)
        else:
            with open(f'{name}.pkl', 'wb') as data_file:
                pickle.dump(data, data_file, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(f'{name}.npy', data)
        print("Сохранение завершено")

    def load(self, data_dir: str):
        """
        Загрузка данных формата pkl и npy
        :param data_dir: путь до данных
        :return: загруженные данные
        """
        if 'pkl' in data_dir:
            with open(f'{data_dir}', 'rb') as data_file:
                data = pickle.load(data_file)
                return data
        elif 'npy' in data_dir:
            data = np.load(f'{data_dir}', allow_pickle=True)
            return data
        else:
            raise Exception("Ошибка при загрузке данных!")


class ImageAugmentorCV:
    """
    Аугментация данных методами
    """

    def __init__(self):
        pass

    def __repr__(self):
        return f'ImageAugmentorOpenCV'

    def open_image(self, image_path: str) -> np.ndarray:
        """
        Чтение изображения
        :param image_path: путь изображения
        :return: изображение в виде матрицы
        """
        return cv2.imread(f"{image_path}")

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

    def gray_scale(self, image: np.ndarray) -> np.ndarray:
        """
        Перевод изображения в ч/б
        :param image: Изображение
        :return: ч/б изображение
        """
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def resized_images(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Изменение размера изображения
        :param image: изображение
        :param width: желаемая ширина изображения
        :param height: желаемая высота изображения
        :return:
        """
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    def show_cv(self, image: np.ndarray, windows_name: str = 'Image'):
        """
        Отображение изображения
        :param image: Изображение
        :param windows_name: Имя окна вывода
        :return: Отображение изображения
        """
        return cv2.imshow(windows_name, image)

    def save_image(self, image: np.ndarray, save_path: str) -> None:
        """
        Сохранение изображения по указанному пути
        :param image: Изображение
        :param save_path: Путь для сохранения
        :return: Сохраняет изображение
        """
        cv2.imwrite(save_path, image)


class ImageAugmentorPillow:
    """
    Аугментация данных методами Pillow
    """

    def __init__(self):
        pass

    def __repr__(self):
        return f'ImageAugmentorPillow'

    def open_image(self, image_path: str):
        """
        Чтение изображения
        :param image_path: путь до изображения
        :return:
        """
        return Image.open(image_path)

    def show_pw(self, image):
        """
        Вывод изображения
        :param image: Изображение
        :return: Отображение изображения
        """
        return image.show()

    def resized_images(self, width: int, height: int, images_data: list):
        """
        Изменение размера изображения
        :param width: желаемая ширина изображения
        :param height: желаемая ширина изображения
        :param images_data: список с изображениями
        :return: список изображений с измененными размерами
        """
        new_size = (width, height)  # Задайте желаемый размер

        resized_images = []
        for image in images_data:
            image = Image.fromarray(image).convert("RGB")
            image = image.resize(new_size)
            resized_images.append(np.array(image))
        return resized_images

    def convert_images_to_array(self, images_path: list[str]) -> np.array:
        """
        Перевод изображения в матричный вид
        :param images_path: Список путей до изображения
        :return: список содержащий матрицы
        """
        images_data = []
        for image_path in images_path:
            image = Image.open(image_path)
            image = np.array(image)
            images_data.append(image)
        return images_data

    def show_mat(self, data_images: np.array, labels: int = None):
        """
        Вывод матричного изображения
        :param data_images: изображение в виде матрицы
        :param labels: метка если имеется, если изображение из датасета и имеет метку
        :return:
        """
        plt.imshow(data_images)
        if labels != None:
            plt.title(f"Метка: {labels}")
        else:
            plt.title("Вывод изображения")
        plt.show()

    def save_image(self, image, save_path: str, name: str = f'//test.png'):
        """
        Сохранение изображения по указанному пути
        :param image: Изображение
        :param save_path: Путь для сохранения
        :param name имя файла
        :return: Сохраняет изображение
        """
        if (save_path[-2:] or name[:2]) == '//':
            image.save(save_path + name)
        elif (save_path[-2:] and name[:2]) == '//':
            raise Exception("Неверно указан путь!")
            name = name[2:]
            image.save(save_path + name)
        else:
            save_path += "//"
            image.save(save_path + name)

    def rotate(self, image, angle: int):
        """
        Вращение изображения на заданный угол [angle]
        :param image: Изображение
        :param angle: Угол вращения
        :return: изображение, перевернутое на заданный угол
        """
        return image.rotate(angle, expand=True)

    def flip(self, image, flip_type: int):
        """
        Отражение (переворот)
        :param image: изображение
        :param flip_type: тип отражения
        0:=> Отражение вокруг вертикальной оси.
        1:=> Отражение вокруг горизонтальной оси.
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
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    def adjust_contrast(self, image, factor: float):
        """
        1.2 и 0.4
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

    def gray_scale(self, image):
        """
        Перевод изображения в ч/б
        :param image: изображение
        :return: ч/б изображение
        """
        return image.convert("L")

    def color_ench(self, image, cor=1.5):
        color_enhancer = ImageEnhance.Color(image)
        return color_enhancer.enhance(cor)


    def run_augmentor(self, file_manager: FileManager):
        """
        Запуск основных методов аугментации данных, по отдельности как для mask, так и для orig.
        :param file_manager: менеджер файлов
        :return:
        """
        # ToDo Да понимаю, требует рефакторинга
        # аугментация orig
        if len(file_manager.path_dir_list_orig) > 0:
            file_manager.len_name = 1
            for image_path in file_manager.path_dir_list_orig:
                list_aug_imgs = []
                img = self.open_image(image_path)
                # img = img.resize((640, 640))
                list_aug_imgs.extend(
                    [img, self.gray_scale(img), self.blur(img), self.add_noise(img), self.adjust_brightness(img, 5.0),
                     self.adjust_brightness(img, -5.0), self.adjust_contrast(img, 0.4),
                     self.adjust_contrast(img, 1.2), self.color_augmentation(img)])
                list_aug_imgs.extend([self.color_augmentation(img, beta_range=-80), self.color_augmentation(img, beta_range=80),
                                      self.color_augmentation(img, alpha_range=-0.9), self.color_augmentation(img, alpha_range=0.9)])
                list_aug_imgs.extend([img, self.color_ench(img, cor=2), self.color_ench(img, cor=-1.0)])
                # list_aug_imgs.extend([img, self.gamma_adjustment(img, gamma=0.7), self.gamma_adjustment(img, gamma=1.8)])
                # list_aug_imgs.extend([img, self.adjust_contrast(img, factor=0.7), self.adjust_contrast(img, factor=1.5)])
                for image in list_aug_imgs:
                    image = image.resize((640, 640))
                    self.save_image(image, save_path=file_manager.new_path_dir_orig,
                                    name=file_manager.new_name_orig + str(
                                        file_manager.len_name) + file_manager.extention)
                    file_manager.len_name += 1
                # for l_img in list_aug_imgs:
                #     rotate_list = []
                #     rotate_list.extend(
                #         [self.rotate(l_img, 45), self.rotate(l_img, -45)])
                #     for r_img in rotate_list:
                #         r_img = r_img.resize((640, 640))
                #         self.save_image(r_img, save_path=file_manager.new_path_dir_orig,
                #                         name=file_manager.new_name_orig + str(
                #                             file_manager.len_name) + file_manager.extention)
                #         file_manager.len_name += 1

        # аугментация mask
        if len(file_manager.path_dir_list_mask) > 0:
            file_manager.len_name = 1
            for image_path in file_manager.path_dir_list_mask:
                list_aug_m_imgs = []
                img = self.open_image(image_path)
                # img = img.resize((640, 640))
                list_aug_m_imgs.extend(
                    [img for _ in range(len(list_aug_imgs))])
                for image in list_aug_m_imgs:
                    image = image.resize((640, 640))
                    self.save_image(image, save_path=file_manager.new_path_dir_mask,
                                    name=file_manager.new_name_mask + str(
                                        file_manager.len_name) + file_manager.extention)
                    file_manager.len_name += 1
                # for l_img in list_aug_m_imgs:
                #     rotate_list = []
                #     rotate_list.extend(
                #         [self.rotate(l_img, 45), self.rotate(l_img, -45)])
                #     for r_img in rotate_list:
                #         r_img = r_img.resize((640, 640))
                #         self.save_image(r_img, save_path=file_manager.new_path_dir_mask,
                #                         name=file_manager.new_name_mask + str(
                #                             file_manager.len_name) + file_manager.extention)
                #         file_manager.len_name += 1
