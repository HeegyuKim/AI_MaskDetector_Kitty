import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

if not cap.isOpened():
    print("Camera not found!")
    sys.exit()

while True:
    ret, frame = cap.read()

    if ret:
        img = frame
        # img = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
        # resizeImg = cv2.resize(src, (width , height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = img[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex,ey,ew,eh) in eyes:
            #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow("frame", img)

        if cv2.waitKey(30) == 27:
            break
    else:
        print("error")

cap.release()
cv2.destroyAllWindows()
