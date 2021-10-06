import argparse
import os
import cv2
from mask_detector import MaskDetector, OpenCVFaceDetector, FacenetDetector, MaskedFaceDrawer


def detect_video(input_file, output_file):
    print(input_file)
    
    in_cap = cv2.VideoCapture(input_file)
    if not in_cap.isOpened(): 
        print(f"파일을 열 수 없습니다: {input_file}")
        exit(0)
        
    width  = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(in_cap.get(cv2.CAP_PROP_FPS))
    length = int(in_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print('width, height, fps, length :', width, height, fps, length)
    
    out_cap = cv2.VideoWriter(output_file, 0x7634706d, int(fps), (int(width), int(height)))
    
    
    index = 0
    
    while True:
        ret, frame = in_cap.read()
        if not ret:
            break
    
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask_drawer.rectangle_faces(image)
        out_cap.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        index += 1
        if index % (fps * 5) == 0:
            print(f"progressing {index} / {length}")
            
    in_cap.release()
    out_cap.release()
    

if __name__ == "__main__":
    mask_detector_model_path = "./resource/model/model.h5"
    opencv_model_path = './resource/opencv/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    opencv_config_path = './resource/opencv/deploy.prototxt'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="분석할 동영상 파일 혹은 디렉터리를 입력하세요")
    parser.add_argument('output_file', help="분석 결과를 저장할 파일 혹은 디렉토리를 입력하세요.")
    parser.add_argument('--detector', default="facenet", help="분석에 사용할 감지기를 고르세요(facenet, opencv)")
    res = parser.parse_args()
        
    mask_detector = MaskDetector(mask_detector_model_path)
    face_detector = FacenetDetector() if res.detector == "facenet" \
        else OpenCVFaceDetector(opencv_model_path, opencv_config_path) # opencv
    mask_drawer = MaskedFaceDrawer(mask_detector, face_detector)
    
    if os.path.isdir(res.input_file): 
        for file in os.listdir(res.input_file):
            if not os.path.exists(res.output_file):
                os.makedirs(res.output_file)
            detect_video(os.path.join(res.input_file, file), os.path.join(res.output_file, file))
    else:
        dir = os.path.dirname(os.path.abspath(res.output_file))
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        detect_video(res.input_file, res.output_file)
        