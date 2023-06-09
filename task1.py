import os
import cv2

print('Task 1. OpenCV version: ' + cv2.__version__)

raw_image = image = cv2.imread('friends.jpg')

# Зменшення зображення зі збереженням відношення сторін
h, w = image.shape[0:2]  # висота і ширина оригінального зображення
h_new = 800  # висота нового зображення
ratio = w / h  # відношення ширини до висоти
w_new = int(h_new * ratio)  # ширина нового зображення
image = cv2.resize(image, (w_new, h_new))  # зміна розміру оригінального зображення

# Виявлення обличчя, усмішки, очей, людей
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
# left_ear_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_leftear.xml')
# right_ear_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_rightear.xml')

gray_filter = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_filter, 1.07, 3)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray_filter[y: y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]

    smile = smile_cascade.detectMultiScale(roi_gray, 1.25, 17)
    for (sx, sy, sw, sh) in smile:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)

    eye = eye_cascade.detectMultiScale(roi_gray, 1.04, 10)
    for (ex, ey, ew, eh) in eye:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)

    nose = nose_cascade.detectMultiScale(roi_gray, 1.15)
    for (nx, ny, nw, nh) in nose:
        cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 255), 1)

    # left_ear = left_ear_cascade.detectMultiScale(roi_gray, 1.15)
    # right_ear = left_ear_cascade.detectMultiScale(roi_gray, 1.15)
    # for (nx, ny, nw, nh) in left_ear + right_ear:
    #     cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 255), 1)

os.makedirs('output', exist_ok=True)
cv2.imwrite('output/task1-output.jpg', image)

cv2.imshow('Task1', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Підрахунок кількості людей на фото
scaling_factor = 0.5
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
resized = cv2.resize(raw_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
people_rects = hog.detectMultiScale(resized, winStride=(8, 8), padding=(30, 30), scale=1.06)
for (x, y, w, h) in people_rects[0]:
    cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
print(f'На фото знайдено {len(people_rects[0])} людей!')
