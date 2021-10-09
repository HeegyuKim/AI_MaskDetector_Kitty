# 이 예제는 원본 학습 데이터에서 학습에 필요한 얼굴만을 떼내서 따로 저장하는 예제입니다.
import os
import cv2
from tqdm import tqdm
from mask_detector import OpenCVFaceDetector


face_detector = OpenCVFaceDetector()

# 디렉토리 내의 사진에서 얼굴만을 추출해서 반환합니다.
def load_dataset(dirname):
    for i in os.listdir(dirname):
        img_path = os.path.join(dirname, i)
        faces, _, _ = face_detector.detect_faces_from_file(img_path)

        for face in faces:
            yield cv2.cvtColor(face, cv2.COLOR_RGB2BGR)


# with_mask, without_mask 내의 사진에서 얼굴을 0_mask, 1_face 라는 디렉토리에 저장합니다.
src_target_list = [
    ("./dataset/with_mask", "./dataset/train/0_mask"),
    ("./dataset/without_mask", "./dataset/train/1_face"),
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
