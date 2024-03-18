"""
    Модуль отрезания пальцев.
"""
from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from matplotlib.patches import Polygon

# create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="lgPDkrQGntDDfpcouQNN"
)
img = '00528.png'
# run inference on a local image
# print(CLIENT.infer("test1.png", model_id="segfinger/2"))
result = CLIENT.infer(img, model_id="segfinger/2")
# class Countours:
#     countours_list = []
#     def __init__(self):
#         self.x = []
#         self.y = []
#
# for pred in result['predictions']:
#     print(pred)
#     cont = Countours()
#     Countours.countours_list.append(cont)
#     for xy in pred['points']:
#         cont.x.append(xy['x'])
#         cont.y.append(xy['y'])
#
# for cnt in Countours.countours_list:
#     cnt: Countours
#     cnt.x.append(cnt.x[0])
#     cnt.y.append(cnt.y[0])
#
# plt.plot(Countours.countours_list[0].x, Countours.countours_list[0].y, '-r')
# plt.plot(Countours.countours_list[1].x, Countours.countours_list[1].y, '-g')
# plt.plot(Countours.countours_list[2].x, Countours.countours_list[2].y, '-b')
# plt.show()
# Загрузка изображения
# image_path = "test3.png"
image = np.array(Image.open(img))

# Отображение изображения
plt.imshow(image)

# Отображение контуров
for pred in result['predictions']:
    points = [(point['x'], point['y']) for point in pred['points']]
    polygon = Polygon(points, edgecolor='black', linewidth=2, fill=True, facecolor='black')
    plt.gca().add_patch(polygon)

# # Закраска областей внутри контуров
# for pred in result['predictions']:
#     points = [(point['x'], point['y']) for point in pred['points']]
#     polygon = Polygon(points, closed=True, facecolor='none')
#     plt.gca().add_patch(polygon)

plt.axis('off')
plt.savefig("result.png", bbox_inches='tight', pad_inches=0)
plt.show()