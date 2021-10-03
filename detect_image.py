import argparse
import cv2
import os
from mask_detector import MaskDetector, FacenetDetector, OpenCVFaceDetector, MaskedFaceDrawer


mask_detector_model_path = "./resource/model/model.h5"
opencv_model_path = './resource/opencv/res10_300x300_ssd_iter_140000_fp16.caffemodel'
opencv_config_path = './resource/opencv/deploy.prototxt'

parser = argparse.ArgumentParser()
parser.add_argument('input_file', help="분석할 이미지 파일 혹은 디렉터리를 입력하세요")
parser.add_argument('output_file', help="분석 결과를 저장할 파일 혹은 디렉토리를 입력하세요.")
parser.add_argument('--detector', default="facenet", help="분석에 사용할 감지기를 고르세요(facenet, opencv)")
res = parser.parse_args()
    
mask_detector = MaskDetector(mask_detector_model_path)
face_detector = FacenetDetector() if res.detector == "facenet" \
    else OpenCVFaceDetector(opencv_model_path, opencv_config_path) # opencv
mask_drawer = MaskedFaceDrawer(mask_detector, face_detector)


def detect_image(input_file, output_file):
    print(input_file)
    
    image = cv2.imread(input_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask_drawer.rectangle_faces(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file, image)
    

if __name__ == "__main__":
    if os.path.isdir(res.input_file): 
        for file in os.listdir(res.input_file):
            if not os.path.exists(res.output_file):
                os.makedirs(res.output_file)
            detect_image(os.path.join(res.input_file, file), os.path.join(res.output_file, file))
    else:
        dir = os.path.dirname(os.path.abspath(res.output_file))
        if not os.path.exists(dir):
            os.makedirs(dir)
            detect_image(res.input_file, res.output_file)