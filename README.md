# Mask Detector
Mask Detector는 사진과 영상에서 마스크를 착용하지 않은 사람을 찾아 표시해주는 파이썬 어플리케이션 및 API를 제공합니다.<br/>

[main](./resource/readme/main.png)
cottonbro님의 동영상, 출처: Pexels<br/>
Developed by [김영수(Young-Soo-Kim)](https://github.com/Young-Soo-Kim), [김희규(HeegyuKim)](https://github.com/HeegyuKim)

## Dependencies
[License List](https://github.com/osamhack2021/AI_MaskDetector_Kitty/blob/master/DEPENDENCIES)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [Pytorch](https://github.com/pytorch/pytorch)
- [Scikit-Learn](https://github.com/scikit-learn/scikit-learn)
- [Numpy](https://github.com/numpy/numpy)
- [Matplotlib](https://github.com/matplotlib/matplotlib)
- [Python Image Library](https://github.com/python-pillow/Pillow)
- [opencv-python](https://github.com/opencv/opencv-python)
- [Pytest](https://github.com/pytest-dev/pytest)
- [PyQT5](https://www.riverbankcomputing.com/software/pyqt/)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
## Installation
먼저 저장소를 clone 한 뒤, 패키지 매니저 [pip](https://pip.pypa.io/en/stable/) 를 이용해서 필요한 라이브러리를 설치합니다.

```bash
$ git clone https://github.com/osamhack2021/AI_MaskDetector_Kitty.git
$ cd AI_MaskDetector_Kitty
$ pip install -r requirements.txt
```

## GUI 어플리케이션 사용방법
카메라 혹은 동영상 파일을 선택하여 기능을 사용해볼 수 있는 GUI 어플리케이션은 run_app.py를 실행하여 사용할 수 있습니다.
```bash
> python run_app.py
```
![GIF](./resource/readme/readme_info_02.gif)<br/>

실시간 카메라 버튼을 클릭하여 연결된 카메라로부터 마스크 탐지를 하거나 동영상 파일에서 마스크 팀지가 가능합니다.

## API 사용방법
구체적인 클래스와 각 메서드의 기능에 대해서 알아보고 싶다면 [API 문서](./docs/API.md)를 참고하세요.<br/>
### 사진에서 마스크를 쓴 얼굴을 찾기
전체 예제는 [예제](examples/detect_image_masked_face.py)를 참고하세요.
1. 분석할 이미지를 로드합니다
```python3
import cv2
image_path = "./resource/sample/image/pexels-gustavo-fring-4127449.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```
2. FacenetDetector 혹은 OpenCVFaceDetector를 이용해서 이미지에서 얼굴을 찾습니다
```python3
from mask_detector import FacenetDetector, OpenCVFaceDetector
face_detector = FacenetDetector()
# face_detector = OpenCVFaceDetector()

faces, confidences, boxes = face_detector.detect_faces(image)
```
3. MaskDetector를 이용해서 찾은 얼굴 이미지가 마스크를 썼는지 판별합니다.
```python3
from mask_detector import MaskDetector
mask_detector = MaskDetector()
mask_probs = mask_detector.predict(faces)

print("마스크 쓴 확률", mask_probs) 
> 마스크 쓴 확률 [0.99954575 0.99999905 0.9970963  0.9999945 ]
```
4. 마스크 쓴 확률이 0.5 이상일 경우 마스크를 쓰지 않았다고 가정하고 마스크를 쓴 사람이 몇명인지 판단합니다.
```python3
mask_count = sum(1 for p in mask_probs if p >= 0.5)
print("마스크 쓴 사람은 총 {}명입니다.".format(mask_count))
> 마스크 쓴 사람은 총 4명입니다
```

### MaskedFaceDrawer를 이용하여 사진에서 마스크 쓴 영역에 그림그리기
```python3
from mask_detector import MaskedFaceDrawer
mask_drawer = MaskedFaceDrawer(mask_detector, face_detector)

mask_drawer.rectangle_faces(image)
```
MaskedFaceDrawer는 마스크를 썼다고 판단되는 얼굴 영역에는 초록 사각형을 그리며, 쓰지 않았다고 판단되는 얼굴 영역에는 붉은 사각형을 그립니다. 얼굴 위에는 얼굴 확신도와 마스크 착용 확률을 표시합니다.

#### 결과물 예시
![사진1](resource/readme/detected-yoav-aziz-T4ciXluAvIE-unsplash.jpg)<br/>
Photo by <a href="https://unsplash.com/@yoavaziz?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Yoav Aziz</a> on <a href="https://unsplash.com/@yoavaziz?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a><br/>
![사진2](resource/readme/detected-victor-he-UXdDfd9ma-E-unsplash.jpg)<br/>
Photo by <a href="https://unsplash.com/@victorhwn725?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Victor He</a> on <a href="https://unsplash.com/s/photos/mask?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a><br/>
  
### [detect_image.py](detect_image.py)를 사용하기
파이썬 코드를 사용하지 않고 커맨드 명령을 이용해 기능을 활용해보는 방법도 있습니다.

파일 하나 
```
> python -m mask_detector.detect_image resource/sample/image/pexels-gustavo-fring-4127449.jpg detected.jpg
```
<br/>
폴더 내 모든 파일
```
# images/ 폴더에 있는 파일들의 분석결과가 images-detected/ 에 저장됩니다
> python -m mask_detector.detect_image resource/sample/image/ images-detected/
```
<br>

디텍터로 openCV를 사용
```
# detector에 opencv를 쓰고싶다면
> python -m mask_detector.detect_image resource/sample/image/ images-detected/ --detector=opencv
```


## 동영상에서 마스크 쓴 얼굴을 찾아보기
1. OpenCV를 이용하여 동영상 혹은 카메라에서 불러올 준비를 합니다.
```python3
import cv2

in_cap = cv2.VideoCapture(0) # 카메라에서 불러온다면
# in_cap = cv2.VideoCapture("resource/sample/video/pexels-rodnae-productions-8363849.mp4") # 파일에서 불러온다면

if not in_cap.isOpened(): 
    print(f"파일을 열 수 없습니다: {input_file}")
    exit(0)
```
2. 분석에 필요한 MaskDetector와 FacenetDetector를 생성합니다.
```python3
from mask_detector import FacenetDetector, MaskDetector
mask_detector = MaskDetector()
face_detector = FacenetDetector()
```
3. 프레임을 읽어와서 이미지를 분석할 때와 동일하게 사용합니다.
```python3
ret, frame = in_cap.read()
if ret:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces, confidences, boxes = face_detector.detect_faces(image)
    mask_probs = mask_detector.predict(faces)
```

### 영상에 마스크 쓴 사람과 안쓴 사람을 찾아 표시하고 저장하기

OpenCV와 MaskedFaceDrawer를 활용한다면 가능합니다. [동영상에서 마스크 쓴 얼굴을 찾아서 표시하고 저장하는 예제](examples/detect_video_masked_face.py)를 참고하여 직접 결과물을 만들거나, detect_video.py를 이용하여 기능을 확인할 수 있습니다.<br/><br/>

```
# video.mp4를 읽어서 분석 후 결과를 video-detected.mp4에 저장합니다.
> python -m mask_detector.detect_video resource/sample/video/pexels-rodnae-productions-8363849.mp4 video-detected.mp4

# videos/ 폴더에 있는 파일들의 분석결과가 videos-detected/ 에 저장됩니다
> python -m mask_detector.detect_video resource/sample/video/ videos-detected/

# detector에 opencv를 쓰고싶다면
> python -m mask_detector.detect_video resource/sample/video/ videos-detected/ --detector=opencv
```
#### 결과물 예시
![GIF](./resource/readme/pexels-george.gif)<br/>
George Morina님의 동영상, 출처: Pexels<br/>
![GIF](./resource/readme/test6.gif)<br/>
Everett Bumstead님의 동영상, 출처: Pexels<br/>

## 모델 학습하기
새로운 데이터를 확보하거나 다른 모델로 마스크 분류기를 직접 학습하고 싶을 경우 아래 문서와 예제를 참고한다면 쉽고 빠르게 학습할 수 있습니다

[모델학습 안내문서](./docs/TRAINING.md)

## Datasets
- LFW Face Database: http://vis-www.cs.umass.edu/lfw/
- Real-World Masked Face Dataset, RMFD: https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset

##  Github Repository 내 저장소명 규칙
- Github Repository 내의 모든 코드는 Front-end 임을 명시 합니다.<br/>
https://bre.is/jxkd4o6V
## License
[MIT License](./LICENSE.md)
