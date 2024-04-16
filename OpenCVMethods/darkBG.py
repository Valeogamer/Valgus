import cv2
import numpy as np
import os

in_path = "C://Users//Valentin//Desktop//538RGBMASKS//"
in_path_imgs = os.listdir(in_path)
out_path_imgs = "C://Users//Valentin//Desktop//538KeyPoint//"

background_color = (0, 0, 0)

for img in in_path_imgs:
    seg_img = cv2.imread(in_path + img)
    background_color = (0, 0, 0)
    background_mask = np.all(seg_img == [0, 0, 0], axis=-1)
    seg_img[background_mask] = background_color
    cv2.imwrite(out_path_imgs+img, seg_img)