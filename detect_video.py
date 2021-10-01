import argparse

import cv2
from mask_detector import MaskDetector, OpenCVFaceDetector, FacenetDetector, MaskedFaceDrawer


input_file = './demoVideo/production ID_5115197.mp4'
output_file = './demoVideo/production ID_5115197_output.mp4'

mask_detector_model_path = "./model.h5"
opencv_model_path = './res10_300x300_ssd_iter_140000_fp16.caffemodel'
opencv_config_path = './deploy.prototxt'

mask_detector = MaskDetector(mask_detector_model_path)
# face_detector = OpenCVFaceDetector(opencv_model_path, opencv_config_path)
face_detector = FacenetDetector()
mask_drawer = MaskedFaceDrawer(mask_detector, face_detector)

def main(input_file, output_file):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="분석할 동영상 파일을 입력하세요")
    parser.add_argument('output_file', help="분석 결과를 저장할 동영상 파일을 입력하세요.")
    res = parser.parse_args()
    main(**vars(res))