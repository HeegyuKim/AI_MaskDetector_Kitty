
import os
import cv2
from tqdm import tqdm
from mask_detector import OpenCVFaceDetector


face_detector = OpenCVFaceDetector()

def load_dataset(dirname):
    for i in os.listdir(dirname):
        img_path = os.path.join(dirname, i)
        faces, _, _ = face_detector.detect_faces_from_file(img_path)
        
        for face in faces:
            yield cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    
        
src_target_list = [
    ("./dataset/with_mask", "./dataset/train/0_mask"),
    ("./dataset/without_mask", "./dataset/train/1_face")
    ]
    
i = 1
for src, target in src_target_list:
    print("{}로부터 학습 데이터 생성 시작".format(src))
    if not os.path.exists(target):
        os.makedirs(target)
        
    for face in tqdm(load_dataset(src)):
        file = os.path.join(target, "{}.jpg".format(i))
        cv2.imwrite(file, face)
        i += 1