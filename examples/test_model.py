import os
import matplotlib.pyplot as plt
from mask_detector import MaskDetector, FacenetDetector


model_path = "resource/model/model.h5"
test_dir = "./resource/sample/image/"
mask_detector = MaskDetector(model_path)
face_detector = FacenetDetector()

# 테스트로 쓸 파일 경로들을 가져옵니다.
test_files = [os.path.join(test_dir, image_file) for image_file in os.listdir(test_dir)]


# FaceDetector를 이용해서 파일에서 얼굴을 찾고
# 학습한 모델을 MaskDetector로 불러와서 마스크를 썼는지 예측합니다.
faces, probs = [], []
for image_file in test_files:
    image_faces, _, _ = face_detector.detect_faces_from_file(image_file)
    
    if len(image_faces) > 0:
        image_probs = mask_detector.predict(image_faces)
        
        faces.extend(image_faces)
        probs.extend(image_probs)
    
    
# 예측 결과를 표시합니다.
len_faces = len(faces)
plt.figure(figsize=(15,15))

for i, face, prob in zip(range(len(faces)), faces, probs):
    plt.subplot(len_faces // 10 + 1,10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(face / 255.)

    if prob > 0.5:
        label = 'Mask {:.2f}'.format(prob.item())
    else:
        label = 'Face {:.2f}'.format(prob.item())

    plt.xlabel(label) 
        
plt.savefig("test.png")