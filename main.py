from DataProcessing import FileManager, CreateDatasetImages, ImageAugmentorPillow
from sklearn.utils import shuffle
import Constants
import time

if __name__ == '__main__':
    start_time = time.time()
    # Необходимые объекты
    file_manager = FileManager(path_dir=Constants.PATH_DIR, path_dir_p=Constants.PATH_DIR_P,
                               path_dir_o=Constants.PATH_DIR_O, new_n_p=Constants.NAME_P, new_n_o=Constants.NAME_O,
                               extention=Constants.EXTENTION_PNG, new_path_dir=Constants.NEW_PATH_DIR,
                               new_path_dir_o=Constants.NEW_PATH_DIR_O, new_path_dir_p=Constants.NEW_PATH_DIR_P)
    create_dataset = CreateDatasetImages()
    img_aug = ImageAugmentorPillow()

    # список путей до каждого файла (на данном шаге вызов отладочный)
    file_manager.path_dir_list_pronation
    file_manager.path_dir_list_overpronation

    # Аугментация данных.
    img_aug.run_augmentor(file_manager)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Время выполнения:", execution_time, "секунд")
    # переименуем все файлы и поменяем расширение (исходные данные)
    # Note: По идее можно отказать от переименования, она нужна была если бы архитектура НС
    #  строилась с применением маски
    # file_manager.create_path_dir_list()  # обновим список
    # file_manager.rename_rextention(path_dir=file_manager.path_dir_pronation, new_name=file_manager.new_name_pronation,
    #                                extention=file_manager.extention)
    # file_manager.rename_rextention(path_dir=file_manager.path_dir_overpronation,
    #                                new_name=file_manager.new_name_overpronation, extention=file_manager.extention)

    # Этап формирования датасета
    # Note: Пока не совсем понятно какой объект должен конвертировать изображение в матрицу numpy.array
    # Конвертирование изображения в матрицу (numpy.array)
    # file_manager.create_path_dir_list()
    # file_manager.update_result_data()
    # imgs_data_overpronation = img_aug.convert_images_to_array(file_manager.result_path_dir_overpronation)
    # imgs_data_pronation = img_aug.convert_images_to_array(file_manager.result_path_dir_pronation)
    #
    # # Изменение размера изображений
    # imgs_data_overpronation = img_aug.resized_images(256, 256, imgs_data_overpronation)
    # imgs_data_pronation = img_aug.resized_images(256, 256, imgs_data_pronation)

    # Присвоение меток классам
    # overpronation_labels = [0] * len(imgs_data_overpronation)
    # pronation_labels = [1] * len(imgs_data_pronation)
    # data = imgs_data_overpronation + imgs_data_pronation
    # labels = overpronation_labels + pronation_labels
    # data, labels = shuffle(data, labels, random_state=42)
    # create_dataset.save(data=data, name='data')
    # create_dataset.save(data=labels, name='labels')
    # data_n = create_dataset.load('data.npy')
    # data_p = create_dataset.load('data.pkl')
    # labels_n = create_dataset.load('labels.npy')
    # labels_p = create_dataset.load('labels.pkl')
    print()
