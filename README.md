# Mask Detector
이미지와 영상 속에서 사람 얼굴을 찾은 뒤 마스크 착용 여부를 판단해주는 기능을 제공합니다.<br/>

![GIF](./resource/readme/test4.gif)<br/>
cottonbro님의 동영상, 출처: Pexels<br/>

Developed by [김영수(Young-Soo-Kim)](https://github.com/Young-Soo-Kim), [김희규(HeegyuKim)](https://github.com/HeegyuKim)

## Installation

패키지 매니저 [pip](https://pip.pypa.io/en/stable/) 를 이용해서 설치합니다.

```bash
pip install -r requirements.txt
pip install facenet-pytorch # FacenetDetector를 사용하려면 설치
pip install wandb # 학습에 wandb를 사용하려면 설치
```

## Usage
### 학습된 모델 사용하기
#### 사진에서 얼굴 찾아서 표시하고 저장하기

![사진1](resource/readme/detected-yoav-aziz-T4ciXluAvIE-unsplash.jpg)<br/>
Photo by <a href="https://unsplash.com/@yoavaziz?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Yoav Aziz</a> on <a href="https://unsplash.com/@yoavaziz?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a><br/>
![사진2](resource/readme/detected-victor-he-UXdDfd9ma-E-unsplash.jpg)<br/>
Photo by <a href="https://unsplash.com/@victorhwn725?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Victor He</a> on <a href="https://unsplash.com/s/photos/mask?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a><br/>
  
1. [detect_image.py](detect_image.py)를 사용하기
```
# 파일 하나를 분석해서 저장함.
> python3 detect_image.py image.jpg image-detected.jpg

# demoImage/ 폴더에 있는 파일들의 분석결과가 demoImage-detected-facenet/ 에 저장됩니다
> python3 detect_image.py demoImage/ demoImage-detected-facenet/

# detector에 opencv를 쓰고싶다면
> python3 detect_image.py demoImage/ demoImage-detected-ocv/ --detector=opencv
```

2. 코드에서 사용하기
```python
import cv2
from mask_detector import MaskDetector, FacenetDetector, OpenCVFaceDetector, MaskedFaceDrawer

mask_detector_model_path = "./model.h5"
opencv_model_path = './res10_300x300_ssd_iter_140000_fp16.caffemodel'
opencv_config_path = './deploy.prototxt'

# opencv 로 이미지를 읽는다. 기본 BGR이므로 RGB로 변경해야 한다.
image = cv2.imread(image_input)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# OpenCVFaceDetector 혹은 FacenetDetector를 이용해서 이미지에서 얼굴을 찾는다
# face_detector = OpenCVFaceDetector(opencv_model_path, opencv_config_path)
# faces, confidences, boxes = face_detector.detect_faces(image)
face_detector = FacenetDetector()
faces, confidences, boxes = face_detector.detect_faces(image)

# faces: 찾은 얼굴 영역의 이미지 (기본 64x64로 변환됨)
# confidences: 찾은 얼굴 영역의 확신도
# boxes: 얼굴 영역의 좌표 리스트 (x1, y1, x2, y2)
print("얼굴 개수, 확률, 영역", faces.shape, confidences, boxes)
#  (2, 64, 64, 3), [0.998097   0.99991643] [[745 104 803 176] [444  71 502 141]]

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
```
  
#### 동영상에서 얼굴 찾아서 표시하고 저장하기

![GIF](./resource/readme/pexels-george.gif)<br/>
George Morina님의 동영상, 출처: Pexels<br/>
![GIF](./resource/readme/test6.gif)<br/>
Everett Bumstead님의 동영상, 출처: Pexels<br/>

1. [detect_video.py](detect_video.py)를 사용하기
```
# video.mp4를 읽어서 분석 후 결과를 video-detected.mp4에 저장합니다.
> python3 detect_video.py video.mp4 video-detected.mp4

# demoImage/ 폴더에 있는 파일들의 분석결과가 demoImage-detected-facenet/ 에 저장됩니다
> python3 detect_video.py demoVideo/ demoVideo-detected-facenet/

# detector에 opencv를 쓰고싶다면
> python3 detect_video.py demoVideo/ demoVideo-detected-ocv/ --detector=opencv
```
2. 코드에서 사용하기
```python
import cv2
from mask_detector import MaskDetector, OpenCVFaceDetector, MaskedFaceDrawer

input_file = './demoVideo/test1.mp4'
output_file = './demoVideo/test1_output.mp4'

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

out_cap = cv2.VideoWriter(output_file, 0x7634706d, fps, (width, height))

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
```

## Training
#### 1. 학습 데이터 추가
train/with_mask/ 폴더에는 마스크를 착용한 얼굴 사진을<br/>
train/without_mask/ 폴더에는 마스크를 착용하지 않은 얼굴 사진을 추가합니다.
#### 2. 학습
```python
> python train.py
Epoch 1/5
65/65 [==============================] - 26s 392ms/step - loss: 0.3580 - accuracy: 0.8483 - val_loss: 0.1343 - val_accuracy: 0.9652
Epoch 2/5
65/65 [==============================] - 25s 388ms/step - loss: 0.1823 - accuracy: 0.9391 - val_loss: 0.1002 - val_accuracy: 0.9783
Epoch 3/5
65/65 [==============================] - 26s 405ms/step - loss: 0.1886 - accuracy: 0.9377 - val_loss: 0.1252 - val_accuracy: 0.9522
Epoch 4/5
65/65 [==============================] - 25s 379ms/step - loss: 0.1460 - accuracy: 0.9483 - val_loss: 0.1308 - val_accuracy: 0.9565
Epoch 5/5
65/65 [==============================] - 25s 383ms/step - loss: 0.1235 - accuracy: 0.9609 - val_loss: 0.1250 - val_accuracy: 0.9609
```
학습된 모델을 model.h5에 저장합니다.

Weight & Biases Logging 를 사용하려면 `wandb login`을 통해 Wandb 계정을 생성합니다.<br/>
사용하지 않으려면 `wandb offline`을 실행해서 wandb를 비활성화합니다.<br/>
자세한 내용은 https://docs.wandb.ai/ 를 참고하세요


#### 3. 결과 테스트
```python
python test.py
```
테스트 결과가 test.png 파일로 저장됩니다.<br/>
![test.png](resource/readme/test.png)


## Datasets
- LFW Face Database: http://vis-www.cs.umass.edu/lfw/
- Real-World Masked Face Dataset，RMFD: https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset