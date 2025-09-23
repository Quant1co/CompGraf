import cv2
import matplotlib.pyplot as plt

# Загрузка изображения (OpenCV читает в формате BGR)
image_path = 'blue_red.png'  # Замените на путь к вашему изображению
img = cv2.imread(image_path)

if img is None:
    print("Ошибка: изображение не найдено или не удалось загрузить.")
    exit()

# Выделение каналов (B, G, R в OpenCV)
blue_channel = img[:, :, 0]  # Синий канал
green_channel = img[:, :, 1]  # Зелёный канал
red_channel = img[:, :, 2]    # Красный канал

# Вывод результатов: показываем каждый канал как grayscale-изображение
cv2.imshow('Red Channel', red_channel)
cv2.imshow('Green Channel', green_channel)
cv2.imshow('Blue Channel', blue_channel)

# Если нужно сохранить каналы в файлы (раскомментируйте):
# cv2.imwrite('red_channel.jpg', red_channel)
# cv2.imwrite('green_channel.jpg', green_channel)
# cv2.imwrite('blue_channel.jpg', blue_channel)

# Построение гистограмм для каждого канала (отдельно)
# Гистограмма для красного канала
plt.figure()
plt.title('Red Channel Histogram')
plt.hist(red_channel.ravel(), bins=256, range=[0, 256], color='red')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()

# Гистограмма для зелёного канала
plt.figure()
plt.title('Green Channel Histogram')
plt.hist(green_channel.ravel(), bins=256, range=[0, 256], color='green')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()

# Гистограмма для синего канала
plt.figure()
plt.title('Blue Channel Histogram')
plt.hist(blue_channel.ravel(), bins=256, range=[0, 256], color='blue')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()

# Ждём нажатия клавиши для закрытия окон изображений
cv2.waitKey(0)
cv2.destroyAllWindows()