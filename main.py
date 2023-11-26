import cv2
import numpy as np

# Відкриття відеофайлу
cap = cv2.VideoCapture('IMG_8335.MOV')

# Опціонально: визначення радіуса вітряка
windmill_radius = 56.0  # припустимий радіус в метрах

# Ініціалізація деяких змінних
prev_frame = None
angular_speeds = []

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Перетворення зображення в градації сірого
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        # Визначення різниці між поточним та попереднім кадром
        diff = cv2.absdiff(gray_frame, prev_frame)

        # Визначення кутової швидкості (простий приклад)
        angular_speed = np.sum(diff) / diff.size
        angular_speeds.append(angular_speed)

        # Відображення поточного кадру
        cv2.imshow('Wind Speed Detection', frame)

    prev_frame = gray_frame

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Розрахунок лінійної швидкості
linear_speeds = [angular_speed * windmill_radius for angular_speed in angular_speeds]
speed_of_wind = np.mean(linear_speeds) / 3.6

# Виведення результатів
print("Average Angular Speed:", np.mean(angular_speeds))
print("Average Linear Speed (Wind Speed):", np.mean(speed_of_wind))
