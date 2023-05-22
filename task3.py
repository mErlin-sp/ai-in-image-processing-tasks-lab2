import os.path

import cv2
import numpy
from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip

print('Task 3. OpenCV version: ' + cv2.__version__)

video_url = 'https://www.youtube.com/watch?v=6FJR2YHcxhI'  # https://www.youtube.com/watch?v=6FJR2YHcxhI
video_file_name = 'oslo-walking-tour.mp4'  # oslo-walking-tour.mp4
video_cropped_file_name = 'oslo-walking-tour-cropped.mp4'
subclip_start_time = 240
subclip_end_time = 300

if not os.path.exists(video_cropped_file_name):
    if not os.path.exists(video_file_name):
        # Завантаження відео з YouTube
        print('Downloading video from YouTube...')
        yt = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
        stream = yt.streams.filter(file_extension='mp4').first()
        stream.download(output_path='./', filename=video_file_name)
        print('Download finished.')

    print('Making 30 second subclip from video...')
    video = VideoFileClip(video_file_name).subclip(subclip_start_time, subclip_end_time)
    video.write_videofile(video_cropped_file_name, fps=20)
    print('Subclip created.')

# Визначення класифікаторів для виявлення пішоходів та облич
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Читання відео з файлу
cap = cv2.VideoCapture(video_cropped_file_name)

# Створення директорії для запису відео
os.makedirs('output', exist_ok=True)

# Визначення кодеку та відеопотоку для запису відео у форматі .mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/task3-output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

skip_frames = 1  # пропустити кожен другий кадр
frame_count = 0

print('Processing frames...')
while cap.isOpened():
    # Зчитування кадру з відео
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    if (frame_count - 1) % 100 == 0:
        print('Processed frames: ' + str(frame_count))

    if frame_count % (skip_frames + 1) == 0:
        # Конвертування кадру в чорно-білий
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Виконання детекції пішоходів та облич на кадрі
        boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4))
        boxes = numpy.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        # pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.02,
                                              minNeighbors=25)  # scaleFactor=1.05, minNeighbors=15

        # Виділення пішоходів та обличів на кадрі
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 1)

        # for (x, y, w, h) in pedestrians:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Запис кадру до відео`
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
print('Processing finished. Result video saved in output folder.')
