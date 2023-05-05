import os.path

import cv2
import numpy
from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip

if not os.path.exists('oslo-walking-tour-cropped.mp4'):
    if not os.path.exists('oslo-walking-tour.mp4'):
        # Завантаження відео з YouTube
        yt = YouTube('https://www.youtube.com/watch?v=6FJR2YHcxhI', use_oauth=True, allow_oauth_cache=True)
        stream = yt.streams.filter(file_extension='mp4').first()
        stream.download(output_path='./', filename='oslo-walking-tour.mp4')

    video = VideoFileClip('oslo-walking-tour.mp4').subclip(200, 230)
    video.write_videofile('oslo-walking-tour-cropped.mp4', fps=20)

# Визначення класифікаторів для виявлення пішоходів та облич
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Читання відео з файлу
cap = cv2.VideoCapture('oslo-walking-tour-cropped.mp4')

# Створення директорії для запису відео
os.makedirs('output', exist_ok=True)

# Визначення кодеку та відеопотоку для запису відео у форматі .mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/task3-output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

skip_frames = 2  # пропустити кожен другий кадр
count = 0

while cap.isOpened():
    # Зчитування кадру з відео
    ret, frame = cap.read()

    if not ret:
        break

    count += 1
    if count % (skip_frames + 1) == 0:
        # Конвертування кадру в чорно-білий
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Виконання детекції пішоходів та облич на кадрі
        boxes, weights = hog.detectMultiScale(frame, winStride=(16, 16))
        boxes = numpy.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        # pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=15)

        # Виділення пішоходів та обличів на кадрі
        for (xa, ya, xb, yb) in boxes:
            cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 255, 0), 1)

        # for (x, y, w, h) in pedestrians:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Запис кадру до відео
        out.write(frame)

        # Показ кадру з відображенням виділених пішоходів та облич
        cv2.imshow('Task3', frame)

    # Вихід з циклу при натисканні клавіші 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Звільнення ресурсів
cap.release()
out.release()
cv2.destroyAllWindows()
