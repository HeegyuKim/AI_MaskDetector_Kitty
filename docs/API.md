# API Documentation
Mask Detector API를 사용하는 방법에 대한 안내입니다.

## MaskDetector API
주어진 이미지 속 얼굴이 마스크를 썼는지를 판별하는 기능을 제공합니다.
#### 생성자
```
MaskDetector(
    model_path="./resource/model/model.h5"
)
```
- model_path: 판별에 사용되는 케라스 모델의 경로이며, 기본값으로 저장소 내의 학습된 모델의 경로가 지정되어 있습니다. 직접 모델을 학습했다면 학습한 다른 모델의 경로를 주어야 합니다.

#### 메서드
##### def predict_one(image)
한 개의 이미지를 입력으로 받아 마스크를 쓴 확률을 0~1범위의 float 값으로 반환합니다.

###### parameters
- image: tf.Tensor 혹은 numpy array로 모델 입력에 맞는 형식이어야 합니다(기본 64, 64, 3). 

###### returns: 
- float, 마스크를 쓴 확률

###### example
```python
image = ... # numpy array (64, 64, 3) shape
p = mask_detector.predict_one(image) # p = 0.86
```
##### def predict(images)
한 개 이상의 이미지 목록을 입력으로 받아 마스크를 쓴 확률을 0~1범위의 float 값으로 반환합니다.

###### parameters
- image: tf.Tensor 혹은 numpy array로 모델 입력에 맞는 4차원의 형식의 1개 이상의 이미지여야 합니다.

###### returns: 
- list(float), 각 이미지들 내의 얼굴이 마스크를 쓴 확률

###### example
```python
images = ... # numpy array (3, 64, 64, 3) shape
p = mask_detector.predict(images) # p = [0.645, 0.112, 0.964]
```

## FaceDetector API

FaceDetector는 사진 내에서 얼굴을 찾아서 반환하는 기능을 추상화한 클래스입니다. 이를 구현하는 두 개의 클래스가 존재하며 사용가능한 메서드는 동일합니다.
###### 서브클래스
- FacenetDetector
- OpenCVFaceDetector

#### FacenetDetector
```
FacenetDetector(
    size=64,
    margin=0,
    device="cpu"
)
```
- size(int): 찾은 얼굴 이미지를 반환할 때 반환될 이미지의 가로, 세로 크기입니다. Facenet은 가로, 세로 크기가 동일한 얼굴 이미지를 반환합니다.
- margin(int): 찾은 얼굴 이미지를 반환할 때 주변 영역을 몇 px 포함할지를 지정합니다. 기본값 0
- device(str): facenet 파이토치 모델이 사용할 계산장치를 지정합니다(기본 `cpu`, GPU를 사용하려면 `cuda`)

#### OpenCVFaceDetector
OpenCV 모델을 이용해서 이미지 내에서 얼굴을 찾아주는 클래스입니다.
```
OpenCVFaceDetector(
    model_path = './resource/opencv/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    config_path = './resource/opencv/deploy.prototxt',
    resize=(64, 64),
    margin=0
)
```
- model_path: OpenCV 모델 파일의 경로
- config_path: OpenCV 모델 Config 파일의 경로
- resize(tuple(int, int)): 찾은 얼굴영역을 반환할 때 반환할 이미지의 가로, 세로 사이즈를 튜플로 지정합니다.
- margin: 반환할 얼굴 이미지에 몇 px 주변 영역을 포함할지 지정합니다.

#### 공통 메서드
##### def detect_faces(image, threshold)
한 개의 이미지를 입력으로 받아 이미지 내에서 얼굴을 찾아서 생성자에서 지정한 크기로 반환합니다.

###### parameters
- image: numpy array로 임의의 가로, 세로 크기를 갖되 채널 수는 3이어야 합니다.
- threshold(float): 찾은 얼굴 영역의 확신도가 threshold값 이하일 경우 반환하지 않도록 합니다. FacenetDetector는 기본값 0.9, OpenCVFaceDetector는 기본값 0.4입니다.

###### returns: 
- list(numpy.array): 찾은 얼굴 영역의 이미지 목록
- list(float): 찾은 얼굴 영역에 대한 확신(confidence) 목록
- list((int, int, int, int)): 찾은 얼굴 영역의 원본 이미지 내의 위치로, 튜플 내의 값은 순서대로 x1, y1, x2, y2 좌표로써 좌상단, 우하단 값을 의미합니다.

###### example
```python
face_detector = OpenCVFaceDetector()
image = ... # numpy array (Width, Height, 3)
faces, probs, boxes = face_detector.detect_faces(image, threshold=0.7)
# faces: np.array(3, 64, 64, 3)
# probs: [0.99, 0.78, 0.85]
# boxes: [(100, 100, 150, 150), (150, 150, 200, 200), (300, 300, 350, 350)]
```
##### def detect_faces_from_file(image, threshold)
`detect_faces` 메서드와 같은 역할을 하지만 이미지가 아닌 파일 경로 문자열을 입력받은 후 이미지를 불러와서 `detect_faces` 메서드를 호출한 값을 반환합니다.
###### parameters
- image_path: 얼굴을 찾을 이미지 파일의 경로
- threshold(float, optional): `detect_faces` 메서드의 threshold 인자로 전달되며 생략시 기본값이 사용됩니다.

###### returns: 
- `detect_faces` 메서드와 동일

###### example
```python
face_detector = OpenCVFaceDetector()
faces, probs, boxes = face_detector.detect_faces_from_file("path_to_image.jpg", threshold=0.7)
# faces: np.array(3, 64, 64, 3)
# probs: [0.99, 0.78, 0.85]
# boxes: [(100, 100, 150, 150), (150, 150, 200, 200), (300, 300, 350, 350)]
```