import cv2
from mask_detector import (
    MaskDetector,
    FacenetDetector,
    OpenCVFaceDetector,
    MaskedFaceDrawer,
)


# 예제로 샘플 이미지를 하나 로드한 뒤 RGB 포맷으로 변경합니다.
image_path = "./resource/sample/image/pexels-gustavo-fring-4127449.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# OpenCVFaceDetector 혹은 FacenetDetector를 이용해서 이미지에서 얼굴을 찾을 수 있습니다
face_detector = FacenetDetector()
# face_detector = OpenCVFaceDetector()

# detect_faces 메서드는 이미지에서 얼굴 영역을 찾아줍니다.
faces, confidences, boxes = face_detector.detect_faces(image)

# faces: 찾은 얼굴 영역의 이미지 (크기를 명시하지 않은 경우 64x64x3)
# confidences: 찾은 얼굴 영역의 확신도(0 ~ 1)
# boxes: 얼굴 영역의 좌표 리스트 (x1, y1, x2, y2)
print("얼굴 개수, 확률, 영역", faces.shape, confidences, boxes, sep="\n")
# 출력결과: 얼굴 개수, 확률, 영역
# (4, 64, 64, 3)
# (0.9996488, 0.9999138, 0.9910159, 0.9999987)
# [[824 287 860 328]
# [645 290 677 331]
# [413 239 444 278]
# [517 347 547 386]]


# MaskDetector는 얼굴사진에서 마스크 착용여부를 검사합니다
# 얼굴 개수만큼 마스크 착용 여부를 확률로 반환합니다.
mask_detector = MaskDetector()
mask_probs = mask_detector.predict(faces)
print("마스크 쓴 확률", mask_probs)
# 출력결과: 마스크 쓴 확률 [0.99954575 0.99999905 0.9970963  0.9999945 ]


# 마스크 쓴 확률이 0.5 이상일 경우 마스크를 착요하지 않았다고 가정할 경우
# 마스크를 쓴 사람이 몇명인지 판단합니다.
mask_count = sum(1 for p in mask_probs if p >= 0.5)
print("마스크 쓴 사람은 총 {}명입니다.".format(mask_count))
# 출력결과: 마스크 쓴 사람은 총 4명입니다
