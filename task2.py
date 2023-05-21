import os
import cv2

print('Task 2. OpenCV version: ' + cv2.__version__)

# Створення об'єкту захоплення відео
cap = cv2.VideoCapture(0)

# Створення об'єкта класифікатора облич
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Час роботи алгоритму в секундах
time_to_run = 30

# Частота кадрів відео
frame_rate = 20

# Розмір відео
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Створення директорії для запису відео
os.makedirs('output', exist_ok=True)

# Створення об'єкта запису відео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/task2-output.mp4', fourcc, frame_rate, (frame_width, frame_height))

# Час початку роботи алгоритму
start_time = cv2.getTickCount()

print('Capturing frames...')
frame_counter = 0
# Поки час роботи алгоритму не минув
while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < time_to_run | frame_counter < (
        time_to_run * frame_rate):
    # Отримання кадру з відео
    ret, frame = cap.read()

    frame_counter += 1
    if (frame_counter - 1) % 100 == 0:
        print('Captured frames: ' + str(frame_counter))

    # Конвертація кадру в чорно-білий
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детектування облич на кадрі
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    # Обчислення координат облич та їх відображення на кадрі
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Запис кадру до відео
    out.write(frame)

    # Показ кадру з відображенням виділених облич
    cv2.imshow('Task2', frame)

    # Вихід з циклу при натисканні клавіші 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Звільнення ресурсів
cap.release()
out.release()
cv2.destroyAllWindows()

print('Capturing finished. Result video saved in output folder.')
