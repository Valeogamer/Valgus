"""
    Модуль определения ключевых точек.
"""
from ultralytics import YOLO
from PIL import Image
model = YOLO('best.pt')

results = model.predict("00537.png", conf=0.20)

print(results)
r = results[0]
im_array = r.plot()  # plot a BGR numpy array of predictions
im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
im.show()