import cv2

# Загрузка изображения
image = cv2.imread('OpenCVMethods/')

# Преобразование в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Бинаризация
_, binary_image = cv2.threshold(gray_image, 25, 100, cv2.THRESH_BINARY)

# Поиск контуров
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Отрисовка контуров
contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

# Отображение изображения с контурами
cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
