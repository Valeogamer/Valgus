"""
    Сегментация ног на изображении.
    UnetModel.
"""
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


model = load_model('UnetModel/unet_model_other_foot.h5')
IMAGE_SIZE = (640, 640)
PLOTS_DPI = 150


def load_test_image(filepath):
    img = tf.io.read_file(filepath)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img / 255.0


def pred_unet(img_path):
    """
    Сегментация изображения
    """
    orig_imgs = [load_test_image(img_path)]
    pred = model.predict(np.array(orig_imgs))
    pred_mask = Binarizer(threshold=0.5).transform(pred.reshape(-1, 1)).reshape(pred.shape)
    # Создаем директорию для сохранения изображений, если её еще нет
    for i in range(len(orig_imgs)):
        # Сохраняем исходное изображение
        # plt.imshow(pred_imgs[i])
        # plt.axis('off')
        # plt.savefig(f'predictions/test_sample_{i + 1}.jpg', bbox_inches='tight', pad_inches=0)
        # plt.close()
        #
        # # Сохраняем мягкую маску
        # plt.imshow(pred[i], cmap='gray')
        # plt.axis('off')
        # plt.savefig(f'predictions/soft_mask_{i + 1}.jpg', bbox_inches='tight', pad_inches=0)
        # plt.close()
        #
        # # Сохраняем бинарную маску
        # plt.imshow(pred_mask[i], cmap='gray')
        # plt.axis('off')
        # plt.savefig(f'predictions/binary_mask_{i + 1}.jpg', bbox_inches='tight', pad_inches=0)
        # plt.close()

        # Сохраняем маскированное изображение
        plt.imshow(orig_imgs[i] * pred_mask[i])
        plt.axis('off')
        plt.savefig(f'predictions/segment_image.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()




# fig, axes = plt.subplots(1, 4, figsize=(15, 18))
# axes = axes.flatten()
#
# for i in range(1):
#     axes[i * 4].imshow(pred_imgs[i])
#     axes[i * 4].grid(False)
#     axes[i * 4].axis(False)
#     axes[i * 4].set_title(f"Test sample #{i + 1}", fontsize=16)
#
#     axes[(i * 4) + 1].imshow(pred[i], cmap='gray')
#     axes[(i * 4) + 1].grid(False)
#     axes[(i * 4) + 1].axis(False)
#     axes[(i * 4) + 1].set_title(f"Soft mask #{i + 1}", fontsize=16)
#
#     axes[(i * 4) + 2].imshow(pred_mask[i], cmap='gray')
#     axes[(i * 4) + 2].grid(False)
#     axes[(i * 4) + 2].axis(False)
#     axes[(i * 4) + 2].set_title(f"Binary mask #{i + 1}", fontsize=16)
#
#     axes[(i * 4) + 3].imshow(pred_imgs[i] * pred_mask[i])
#     axes[(i * 4) + 3].grid(False)
#     axes[(i * 4) + 3].axis(False)
#     axes[(i * 4) + 3].set_title(f"Masked image #{i + 1}", fontsize=16)
#
# # plt.suptitle("Segmentation predictions", fontsize = 24)
# plt.tight_layout(rect=[0, 0, 1, 0.98])
# plt.savefig('predictions.jpg', dpi=PLOTS_DPI, bbox_inches='tight')
# # plt.show()

# import cv2
#
# # Чтение изображения
# image = cv2.imread("predictions/masked_image_1.jpg")
#
# # Преобразование в оттенки серого (если изображение в цвете)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Применение порогового преобразования (например, бинаризации)
# _, binary_image = cv2.threshold(gray_image, 25, 255, cv2.THRESH_BINARY)
#
# # Поиск контуров в бинарном изображении
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Аппроксимация всех контуров
# for contour in contours:
#     epsilon = 0.001 * cv2.arcLength(contour, True) # Здесь 0.01 - это точность аппроксимации, можете изменить по своему усмотрению
#     approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
#
#     # Рисование аппроксимированного контура на исходном изображении
#     cv2.drawContours(image, [approximated_contour], -1, (0, 255, 0), 2)
#
# # Сохранение результата
# cv2.imwrite("output_image.jpg", image)
