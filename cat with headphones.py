import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cat_img = cv2.imread('image.png')

gray = cv2.cvtColor(cat_img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) == 0:
    print("❌ Лицо кота не найдено! Размещаем наушники вручную.")

    x, y, w, h = cat_img.shape[1] // 4, cat_img.shape[0] // 6, cat_img.shape[1] // 2, cat_img.shape[0] // 2
else:
    print("✅ Лицо найдено! Размещаем наушники автоматически.")
    x, y, w, h = faces[0]

headphones_img = cv2.imread('headphones.png', cv2.IMREAD_UNCHANGED)

headphones_resized = cv2.resize(headphones_img, (w + 280, h + 280))

if headphones_resized.shape[2] == 4:
    alpha_mask = headphones_resized[:, :, 3] / 255.0
    headphones_resized = headphones_resized[:, :, :3] 
else:
    alpha_mask = np.ones((h + 40, w + 40), dtype=np.float32)

x_offset = x - 135
y_offset = y - 135

for c in range(3):
    cat_img[y_offset:y_offset + headphones_resized.shape[0], x_offset:x_offset + headphones_resized.shape[1], c] = (
        cat_img[y_offset:y_offset + headphones_resized.shape[0], x_offset:x_offset + headphones_resized.shape[1], c] * (1 - alpha_mask) +
        headphones_resized[:, :, c] * alpha_mask
    )

cv2.imwrite('cat_with_headphones.png', cat_img)
cv2.imshow('Cat with Headphones', cat_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
