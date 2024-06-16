from DataProcessing import FileManager, ImageAugmentorPillow
import Constants
import time

if __name__ == '__main__':
    start_time = time.time()
    file_manager = FileManager(path_dir=Constants.PATH_DIR, path_dir_orig=Constants.PATH_DIR_Orig,
                               path_dir_mask=Constants.PATH_DIR_Mask, new_n_orig=Constants.NAME_Orig,
                               new_n_mask=Constants.NAME_Mask,
                               extention=Constants.EXTENTION_PNG, new_path_dir=Constants.NEW_PATH_DIR,
                               new_path_dir_mask=Constants.NEW_PATH_DIR_Mask,
                               new_path_dir_orig=Constants.NEW_PATH_DIR_Orig)
    img_aug = ImageAugmentorPillow()
    img_aug.run_augmentor(file_manager)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Время выполнения:", f'{execution_time:0.4f}', "секунд")
