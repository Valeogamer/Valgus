# """
#     Модуль отрезания пальцев.
# """
# from inference_sdk import InferenceHTTPClient
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
# from matplotlib.patches import Polygon
#
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="lgPDkrQGntDDfpcouQNN"
# )
#
#
# def pred_no_finger(img_path):
#     result = CLIENT.infer(img_path, model_id="segfinger/2")
#     image = np.array(Image.open(img_path))
#     plt.imshow(image)
#     for pred in result['predictions']:
#         points = [(point['x'], point['y']) for point in pred['points']]
#         polygon = Polygon(points, edgecolor='black', linewidth=2, fill=True, facecolor='black')
#         plt.gca().add_patch(polygon)
#     plt.axis('off')
#     # plt.savefig("predictions/no_finger_img.png", bbox_inches='tight', pad_inches=0)
#     plt.savefig("no_finger_img.png", bbox_inches='tight', pad_inches=0)
#
#
# i_path = "/home/valeogamer/Загрузки/DataTest/00488.png"
# pred_no_finger(i_path)

from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.patches import Polygon

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="lgPDkrQGntDDfpcouQNN"
)

def smooth_contour(points, window_size=5):
    smoothed_points = []
    for i in range(len(points)):
        avg_x = np.mean([points[j][0] for j in range(max(0, i - window_size // 2), min(len(points), i + window_size // 2 + 1))])
        avg_y = np.mean([points[j][1] for j in range(max(0, i - window_size // 2), min(len(points), i + window_size // 2 + 1))])
        smoothed_points.append((avg_x, avg_y))
    return smoothed_points

def pred_no_finger(img_path):
    result = CLIENT.infer(img_path, model_id="segfinger/2")
    image = np.array(Image.open(img_path))
    plt.imshow(image)
    for pred in result['predictions']:
        points = [(point['x'], point['y']) for point in pred['points']]
        smoothed_points = smooth_contour(points, window_size=5)  # Увеличьте размер окна для более сильного сглаживания
        polygon = Polygon(smoothed_points, edgecolor='black', linewidth=2, fill=True, facecolor='black')
        plt.gca().add_patch(polygon)
    plt.axis('off')
    plt.savefig("no_finger_img.png", bbox_inches='tight', pad_inches=0)

i_path = "/home/valeogamer/Загрузки/DataTest/00538.png"
pred_no_finger(i_path)


