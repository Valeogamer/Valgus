import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image = cv2.imread('000128.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    countours, hierarhy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, countours, -1, (0, 255, 0), 2)
    x, y = [], []
    for i in countours:
        for j in i:
            for k in j:
                x.append(k[0])
                y.append(k[-1])
    y_min = min(y)  # минимум по координате Y
    y_max = max(y)  # максимум по координате Y
    y_delta = max(y) - min(y)  # разница между мин макс
    # схема 10 - 30 - 60
    y_1 = int(y_max - y_delta / 10)  # 10%
    y_2 = int(y_max - y_delta * 1 / 3)  # 30%
    y_3 = int(y_max - y_delta * 2 / 3)  # 60%
    # После того как вычилслили оси y_1,2,3, необходимо определить Х, на каждой оси их по 2.
    arrs = []
    arrs.append(gray[y_1])
    arrs.append(gray[y_2])
    arrs.append(gray[y_3])
    x_all_coords = []
    for arr in arrs:
        x_coords = []
        cnt_exit = 0
        left_border = True
        right_border = False
        for i in range(1, len(arr) - 1):
            if left_border:
                # if arr[i - 1] < 10 and arr[i - 1] != 0 and arr[i - 1] != 1 and arr[i + 1] > arr[i]:
                if arr[i] > 10:
                    x_coords.append(i)
                    left_border = False
                    right_border = True
                    cnt_exit += 1
            if right_border:
                # if arr[i - 1] >= arr[i] and (arr[i] == 0 or arr[i] == 1):
                if arr[i] < 10:
                    x_coords.append(i)
                    left_border = True
                    right_border = False
                    cnt_exit += 1
            if cnt_exit >= 4:
                break
        x_all_coords.append(x_coords)
    x_1_l = (x_all_coords[0][0] + x_all_coords[0][1]) / 2
    x_1_r = (x_all_coords[0][2] + x_all_coords[0][3]) / 2

    x_2_l = (x_all_coords[1][0] + x_all_coords[1][1]) / 2
    x_2_r = (x_all_coords[1][2] + x_all_coords[1][3]) / 2

    x_3_l = (x_all_coords[2][0] + x_all_coords[2][1]) / 2
    x_3_r = (x_all_coords[2][2] + x_all_coords[2][3]) / 2
    plt.plot(x, y)
    plt.plot(x_1_l, y_1, 'r*')
    plt.plot(x_1_r, y_1, 'r*')
    plt.plot(x_2_l, y_2, 'g*')
    plt.plot(x_2_r, y_2, 'g*')
    plt.plot(x_3_l, y_3, 'y*')
    plt.plot(x_3_r, y_3, 'y*')
    plt.gca().invert_yaxis()
    plt.imshow(image)
    plt.show()
