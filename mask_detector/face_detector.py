import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import numpy as np


class OpenCVFaceDetector:
    def __init__(self, model_path, config_path, resize=(64, 64)):
        # model = './AI_Mask_Detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
        # config = './AI_Mask_Detector/deploy.prototxt'
        self.net = cv2.dnn.readNet(model_path, config_path)
        self.resize = resize
        
        if self.net.empty():
            raise Exception("opencv face detector net is empty")
            
    def detect_faces(self, image, threshold=0.4, margin=0):
        
        blob = cv2.dnn.blobFromImage(image, 1, (300, 300), (104, 177, 123))
        self.net.setInput(blob)
        
        detect = self.net.forward()
        detect = detect[0, 0, :, :]
        (h, w) = image.shape[:2]
        
        faces = []
        confidences = []
        boxes = []
        
        for i in range(detect.shape[0]):
            confidence = detect[i, 2]
            if confidence < threshold:
                break

            x1 = int(detect[i, 3] * w)
            y1 = int(detect[i, 4] * h)
            x2 = int(detect[i, 5] * w)
            y2 = int(detect[i, 6] * h)

            face = image[y1-margin:y2+margin, x1-margin:x2+margin]

            if self.resize is not None:
                face = cv2.resize(face, self.resize)
                
            faces.append(face)
            confidences.append(confidence)
            boxes.append((x1, y1, x2, y2))
            
        return faces, confidences, boxes
        

class FacenetDetector:
    def __init__(self, size=64, margin=0, device="cpu"):
        self.mtcnn = MTCNN(image_size=size, margin=margin, keep_all=True, post_process=False, device=device)
        
    def detect_faces(self, image, threshold=0.4):
        image = Image.fromarray(image)
        boxes, probs = self.mtcnn.detect(image, landmarks=False)
                
        faces = self.mtcnn.extract(image, boxes, save_path=None)
        faces = torch.permute(faces, (0, 2, 3, 1)).numpy()
        
        boxes = boxes.astype(np.int32) if boxes is not None else []
        return faces, probs, boxes