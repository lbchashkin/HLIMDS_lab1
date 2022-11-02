import cv2

LED_PIN = 23
SERVO_PIN = 25

import sys
import RPi.GPIO as GPIO
from time import sleep

# Инициализация GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Инициализация сервомотра
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

def setAngle(angle):
# Установка угла поворота
    duty=angle/18+2
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

cam = cv2.VideoCapture(0) # видеопоток
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('my_cam_vis.avi', fourcc, 20.0, (640, 480)) # запись видео в файл
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # модель для распознавания

i = 0 
x_old = 0 # старое положение камеры
angle = 90 # начальный угол 90
setAngle(angle)
while True:
    ret, img = cam.read()
    img = cv2.rotate(img, cv2.ROTATE_180)
    if i % 10 == 0:
	# обработка каждого 10 кадра
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # преобразование в чёрно-белый
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) # определение лиц
        if len(faces) == 0: # нет лиц
            GPIO.output(LED_PIN, False) # выключаем светодиод
            angle = 90
            setAngle(angle)
        else: # несколько лиц - выбираем первое
            GPIO.output(LED_PIN, True) # включаем светодиод
            angle = angle - (faces[0][0] - x_old)/10 # изменение угла
            setAngle(angle)
            x_old = faces[0][0]
    for (x, y, w, h) in faces:
		# Отрисовка прямоугольников возле найденных лиц
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('my_cam', img)
    out.write(img)
    i = i + 1
    if cv2.waitKey(10) == 27:
		# Завершение программы по клавише Esc
        break

# Завершение работы
GPIO.cleanup()
cam.release()
out.release()
cv2.destroyAllWindows()
