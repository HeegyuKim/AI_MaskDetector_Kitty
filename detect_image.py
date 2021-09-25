import cv2
from mask_detector import MaskDetector, FacenetDetector, OpenCVFaceDetector, MaskedFaceDrawer

# image_input = "victor-he-UXdDfd9ma-E-unsplash.jpg"
image_input = "yoav-aziz-T4ciXluAvIE-unsplash.jpg"
image_output = "detected-" + image_input

mask_detector_model_path = "./model.h5"
opencv_model_path = './res10_300x300_ssd_iter_140000_fp16.caffemodel'
opencv_config_path = './deploy.prototxt'

# opencv 로 이미지를 읽는다. 기본 BGR이므로 RGB로 변경해야 한다.
image = cv2.imread("demoImage/" + image_input)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# OpenCVFaceDetector를 이용해서 이미지에서 얼굴을 찾는다
# resize 옵션을 이용해서 찾은 얼굴 이미지를 자동으로 64,64로 변경

# face_detector = OpenCVFaceDetector(opencv_model_path, opencv_config_path)
# faces, confidences, boxes = face_detector.detect_faces(image, resize=(64, 64))
face_detector = FacenetDetector()
faces, confidences, boxes = face_detector.detect_faces(image)

# faces: 찾은 얼굴 영역의 이미지
# confidences: 찾은 얼굴 영역의 확신도
# boxes: 얼굴 영역의 좌표 리스트 (x1, y1, x2, y2)
print("얼굴 개수, 확률, 영역", faces.shape, confidences, boxes)
# 2 [0.998097   0.99991643] [[745 104 803 176] [444  71 502 141]]


# MaskDetector를 이용해서 찾은 얼굴 이미지가 마스크를 썼는지 판별
# 얼굴들의 확률를 리턴해줌
mask_detector = MaskDetector(mask_detector_model_path)
mask_probs = mask_detector.predict(faces)
print("마스크 쓴 확률", mask_probs) # 결과 [1. 1.]


# MaskedFaceDrawer는 이미지에서 얼굴을 찾아서 영역에 사각형을 그려준다.
mask_drawer = MaskedFaceDrawer(mask_detector, face_detector)
mask_drawer.rectangle_faces(image)

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite("demoImage/" + image_output, image)