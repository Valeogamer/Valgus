"""
    Модуль отрезания пальцев.
"""
from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.patches import Polygon

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="lgPDkrQGntDDfpcouQNN"
)


def pred_no_finger(img_path):
    img = 'masked_image_1.jpg'
    result = CLIENT.infer(img, model_id="segfinger/2")
    image = np.array(Image.open(img))
    plt.imshow(image)
    for pred in result['predictions']:
        points = [(point['x'], point['y']) for point in pred['points']]
        polygon = Polygon(points, edgecolor='black', linewidth=2, fill=True, facecolor='black')
        plt.gca().add_patch(polygon)
    plt.axis('off')
    plt.savefig("predictions/no_finger_img.png", bbox_inches='tight', pad_inches=0)
