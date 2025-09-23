from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("image.jpg") 
img = img.convert("RGB")
arr = np.array(img, dtype=float)

weights_pal = [0.299, 0.587, 0.114]
gray_pal = np.dot(arr, weights_pal)

weights_hdtv = [0.2126, 0.7152, 0.0722]
gray_hdtv = np.dot(arr, weights_hdtv)

diff = np.abs(gray_pal - gray_hdtv)

Image.fromarray(gray_pal.astype(np.uint8)).save("gray_pal.jpg")
Image.fromarray(gray_hdtv.astype(np.uint8)).save("gray_hdtv.jpg")
Image.fromarray(diff.astype(np.uint8)).save("gray_diff.jpg")

print("Изображения сохранены как gray_pal.jpg, gray_hdtv.jpg и gray_diff.jpg")

plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
plt.imshow(gray_pal.astype(np.uint8), cmap="gray")
plt.title("PAL/NTSC (0.299,0.587,0.114)")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(gray_hdtv.astype(np.uint8), cmap="gray")
plt.title("HDTV (0.2126,0.7152,0.0722)")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(diff.astype(np.uint8), cmap="gray")
plt.title("Разность изображений")
plt.axis("off")

plt.subplot(2,3,4)
plt.hist(gray_pal.ravel(), bins=256, range=(0,255), color="gray")
plt.title("Гистограмма PAL/NTSC")

plt.subplot(2,3,5)
plt.hist(gray_hdtv.ravel(), bins=256, range=(0,255), color="black")
plt.title("Гистограмма HDTV")

plt.tight_layout()
plt.show()
