import cv2
from mask_detector import MaskDetector, OpenCVFaceDetector, MaskedFaceDrawer

input_file = './demoVideo/test1.mp4'
output_file = './movie_output.mp4'

mask_detector_model_path = "./model.h5"
opencv_model_path = './res10_300x300_ssd_iter_140000_fp16.caffemodel'
opencv_config_path = './deploy.prototxt'

in_cap = cv2.VideoCapture(input_file)
if not in_cap.isOpened(): 
    print(f"파일을 열 수 없습니다: {input_file}")
    exit(0)
    
width  = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(in_cap.get(cv2.CAP_PROP_FPS))

print('width, height, fps :', width, height, fps)

out_cap = cv2.VideoWriter(output_file, 0x7634706d, int(fps), (int(width), int(height)))

mask_detector = MaskDetector(mask_detector_model_path)
face_detector = OpenCVFaceDetector(opencv_model_path, opencv_config_path)
mask_drawer = MaskedFaceDrawer(mask_detector, face_detector)

while True:
    ret, frame = in_cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask_drawer.rectangle_faces(image)
    out_cap.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
in_cap.release()
out_cap.release()