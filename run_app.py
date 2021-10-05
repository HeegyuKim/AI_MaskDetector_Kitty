import sys 
from PyQt5 import uic
import PyQt5 
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

from source.opencv.camera_thread import CameraThread

import cv2
import time

form_class = uic.loadUiType("./resource/ui/main_gui.ui")[0] 
class WindowClass(QMainWindow, form_class): 
    th = CameraThread()

    def __init__(self): 
        super().__init__() 
        self.setupUi(self) 

        self.showLogo()

        self.btn_camera.clicked.connect(self.btnCameraClick)
        self.btn_av.clicked.connect(self.btnAvClick)
        self.btn_close.clicked.connect(self.btnCloseClick)
        

    def btnAvClick(self): 
        print("av btn.")
        self.th.terminate()
        
        fname = QFileDialog.getOpenFileName(self, 'Open File', '', 'Video File(*.avi *.mp4 *.mkv);; All File(*)')
        print(fname[0])
        self.play(fname[0])

    def btnCameraClick(self): 
        print("camera btn")

        #region sample code
        #qPixmapVar = QPixmap()
        #qPixmapVar.load('./21.jpg')
        #self.lbl_img.setPixmap(qPixmapVar)
        #endregion
        
        self.th.terminate()
        self.play(0)

    def btnCloseClick(self): 
        print("close btn")
        self.th.terminate()

        print("showLogo()")
        self.showLogo()   

       
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.lbl_img.setPixmap(QPixmap.fromImage(image))

    def play(self, fileName):
        self.th = CameraThread(self)
        self.th.changePixmap.connect(self.setImage)
        self.th.setPlayType(fileName)
        self.th.start()
        self.show()  

    def showLogo(self):
        self.lbl_img.setPixmap(PyQt5.QtGui.QPixmap("./resource/image/logo.png"))


if __name__ == "__main__": 
    app = QApplication(sys.argv) 
    myWindow = WindowClass() 
    myWindow.show() 
    app.exec_()
