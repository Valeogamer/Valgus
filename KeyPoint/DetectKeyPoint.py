"""
    Модуль определения ключевых точек.
"""
from ultralytics import YOLO
from PIL import Image
import numpy as np
model = YOLO('best.pt')

results = model.predict("masked_image_1.jpg", conf=0.40)

for r in results:
    keypoints_tensor = r.keypoints.xy
    keypoints_list = keypoints_tensor.tolist()

    # Получаем координаты ключевых точек для правой ноги
    right_xy = keypoints_list[0][:3]
    r_x, r_y = [], []
    for xy in right_xy:
        r_x.append(xy[0])
        r_y.append(xy[1])

    # Получаем координаты ключевых точек для левой ноги
    left_xy = keypoints_list[1][:3]
    l_x, l_y = [], []
    for xy in left_xy:
        l_x.append(xy[0])
        l_y.append(xy[1])

# Вычисление угла между двумя векторами
def compute_angle(x1, y1, x2, y2):
    dot_product = x1 * x2 + y1 * y2
    magnitude1 = np.sqrt(x1**2 + y1**2)
    magnitude2 = np.sqrt(x2**2 + y2**2)
    angle = np.arccos(dot_product / (magnitude1 * magnitude2))
    return np.degrees(angle)

# Вычисление угла для правой ноги
angle_right_hand = compute_angle(r_x[1] - r_x[0], r_y[1] - r_y[0], r_x[2] - r_x[1], r_y[2] - r_y[1])

# Вычисление угла для левой ноги
angle_left_hand = compute_angle(l_x[1] - l_x[0], l_y[1] - l_y[0], l_x[2] - l_x[1], l_y[2] - l_y[1])

print("Угол между точками для правой руки:", angle_right_hand)
print("Угол между точками для левой руки:", angle_left_hand)
# Вывод изображения
# print(results)
# r = results[0]
# im_array = r.plot()  # plot a BGR numpy array of predictions
# im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
# im.show()