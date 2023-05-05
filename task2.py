import cv2

# Створення об'єкту захоплення відео
cap = cv2.VideoCapture(0)

# Створення об'єкта класифікатора облич
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Час роботи алгоритму в секундах
time_to_run = 30

# Час початку роботи алгоритму
start_time = cv2.getTickCount()

# Поки час роботи алгоритму не минув
while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < time_to_run:
    # Отримання кадру з відео
    ret, frame = cap.read()

    # Конвертація кадру в чорно-білий
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детектування облич на кадрі
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    # Обчислення координат облич та їх відображення на кадрі
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Показ кадру з відображенням виділених облич
    cv2.imshow('frame', frame)

    # Вихід з циклу при натисканні клавіші 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Звільнення ресурсів
cap.release()
cv2.destroyAllWindows()
