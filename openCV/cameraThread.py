import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import sys
import time

class CameraThread(QThread):
    changePixmap = pyqtSignal(QImage)

    model = './res10_300x300_ssd_iter_140000_fp16.caffemodel'
    config = './deploy.prototxt'
    #model = './opencv_face_detector_uint8.pb'
    #config = './opencv_face_detector.pbtxt'

    mask_model = tf.keras.models.load_model('./model.h5')
    probability_model = tf.keras.Sequential([mask_model])
    width = 64
    height = 64

    cap = None
    Running = True

    fileName = 0
    def setPlayType(self, fileName = 0):
        self.fileName = fileName

    def terminate(self):
        print('camera terminate')
        self.Running = False        

        print('camera terminate11')



    def run(self):
        if self.fileName == 0:
            #릴리즈 시에 내부적으로 에러가 발생하는 크래쉬 테스트 (CAP_DSHOW 추가)
            self.cap = cv2.VideoCapture(self.fileName, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.fileName)

        if not self.cap.isOpened():
            print('Camera open failed!')
            #exit()

        net = cv2.dnn.readNet(self.model, self.config)

        if net.empty():
            print('Net open failed!')
            #exit()        

        categories = ['mask','none']
        print('len(categories) = ', len(categories))

        nomaskIcon = cv2.imread('./resource/ui/nomask.png', cv2.IMREAD_UNCHANGED)        

        while self.Running:
            ret, frame = self.cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = img.shape

                blob = cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123))
                net.setInput(blob)
                detect = net.forward()

                detect = detect[0, 0, :, :]

                for i in range(detect.shape[0]):
                    confidence = detect[i, 2]
                    if confidence < 0.4:
                        break

                    x1 = int(detect[i, 3] * w)
                    y1 = int(detect[i, 4] * h)
                    x2 = int(detect[i, 5] * w)
                    y2 = int(detect[i, 6] * h)

                    margin = 0
                    face = img[y1-margin:y2+margin, x1-margin:x2+margin]

                    resize = cv2.resize(face, (self.width , self.height))

                    rgb_tensor = tf.convert_to_tensor(resize, dtype=tf.float32)
                    rgb_tensor /= 255.
                    rgb_tensor = tf.expand_dims(rgb_tensor , 0)

                    # 예측
                    predictions = self.probability_model.predict(rgb_tensor)

                    if predictions[0][0] > predictions[0][1]:# and predictions[0][0] > 0.7:
                        label = 'Mask ' + str(round(predictions[0][0], 3))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    if predictions[0][0] < predictions[0][1]:# and predictions[0][1] > 0.7:
                        # 헤드업 디스플레이 출력
                        fx = (x2 - x1) / nomaskIcon.shape[1] * 0.7
                        cat2 = cv2.resize(nomaskIcon, (0, 0), fx=fx, fy=fx)
                        pos = ((int(x1 + (x2 - x1)*0.15)), int(y1 - (y2 - y1) / 2))
                        self.overlay(frame, cat2, pos)

                        label = 'No Mask ' + str(round(predictions[0][1], 3))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                #cv2.imshow('frame', frame)        

                #PYQT로 이미지 정보 출력
                frame = cv2.resize(frame, dsize=(640,480))
                h, w, ch = frame.shape
                bytesPerLine = ch * w            
                convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_BGR888)
                #p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(convertToQtFormat)
                #self.changePixmap.emit(p)

                if cv2.waitKey(30) == 27:
                    break

            else:
                print('error : ', ret)
                #동영상 실행이 끝났을때 처리
                self.Running = False

        try:
            self.cap.release()
            cv2.destroyAllWindows()	                 
            print('cap release')
        except:
            print('except:')


    #헤드업 디스플레이 출력
    def overlay(self, frame, nomaskIcon, pos):
        if pos[0] < 0 or pos[1] < 0:
            return

        if pos[0] + nomaskIcon.shape[1] > frame.shape[1] or pos[1] + nomaskIcon.shape[0] > frame.shape[0]:
            return

        sx = pos[0]
        ex = pos[0] + nomaskIcon.shape[1]
        sy = pos[1]
        ey = pos[1] + nomaskIcon.shape[0]

        img1 = frame[sy:ey, sx:ex]  # shape=(h, w, 3)
        img2 = nomaskIcon[:, :, 0:3]       # shape=(h, w, 3)
        alpha = 1. - (nomaskIcon[:, :, 3] / 255.)  # shape=(h, w)

        img1[:, :, 0] = (img1[:, :, 0] * alpha + img2[:, :, 0] * (1. - alpha)).astype(np.uint8)
        img1[:, :, 1] = (img1[:, :, 1] * alpha + img2[:, :, 1] * (1. - alpha)).astype(np.uint8)
        img1[:, :, 2] = (img1[:, :, 2] * alpha + img2[:, :, 2] * (1. - alpha)).astype(np.uint8)




if __name__ == "__main__": 
    myWindow = CameraThread() 
    myWindow.run() 
