from DataProcessing import FileManager, CreateDatasetImages, ImageAugmentorPillow
from sklearn.utils import shuffle

if __name__ == '__main__':
    # Необходимые объекты
    file_manager = FileManager()
    create_dataset = CreateDatasetImages()
    img_aug = ImageAugmentorPillow()

    # список путей до каждого файла (на данном шаге вызов отладочный)
    file_manager.path_dir_list_pronation
    file_manager.path_dir_list_overpronation

    # Аугментация данных.
    img_aug.run_augmentor(file_manager)

    # переименуем все файлы и поменяем расширение (исходные данные)
    # Note: По идее можно отказать от переименования, она нужна была если бы архитектура НС
    #  строилась с применением маски
    file_manager.create_path_dir_list()  # обновим список
    file_manager.rename_rextention(path_dir=file_manager.path_dir_pronation, new_name=file_manager.new_name_pronation,
                                   extention=file_manager.extention)
    file_manager.rename_rextention(path_dir=file_manager.path_dir_overpronation,
                                   new_name=file_manager.new_name_overpronation, extention=file_manager.extention)

    # Этап формирования датасета
    # Note: Пока не совсем понятно какой объект должен конвертировать изображение в матрицу numpy.array
    # Конвертирование изображения в матрицу (numpy.array)
    file_manager.create_path_dir_list()
    imgs_data_overpronation = img_aug.convert_images_to_array(file_manager.path_dir_list_overpronation)
    imgs_data_pronation = img_aug.convert_images_to_array(file_manager.path_dir_list_pronation)

    # Изменение размера изображений
    imgs_data_overpronation = img_aug.resized_images(256, 256, imgs_data_overpronation)
    imgs_data_pronation = img_aug.resized_images(256, 256, imgs_data_pronation)

    # Присвоение меток классам
    overpronation_labels = [0] * len(imgs_data_overpronation)
    pronation_labels = [1] * len(imgs_data_pronation)
    data = imgs_data_overpronation + imgs_data_pronation
    labels = overpronation_labels + pronation_labels
    data, labels = shuffle(data, labels, random_state=42)
    create_dataset.save(data=data, name='data')
    create_dataset.save(data=labels, name='labels')
    data_n = create_dataset.load('data.npy')
    data_p = create_dataset.load('data.pkl')
    labels_n = create_dataset.load('labels.npy')
    labels_p = create_dataset.load('labels.pkl')
    print()
