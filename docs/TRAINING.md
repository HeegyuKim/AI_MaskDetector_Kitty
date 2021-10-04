# Training
이 문서에서는 Mask Detector에 필요한 마스크 분류 모델을 학습하는 과정을 안내합니다.<br/>
아래와 같은 과정과 각 과정에 필요한 예제로 이루어져있습니다.
1. 학습 데이터 준비
2. 데이터 전처리
3. 모델 학습
4. 모델 테스트

구체적인 학습 과정을 알고 싶으시다면 예제 코드와 주석을 참고하세요.
## Installation

학습 예제에 필요한 라이브러리를 다운로드 받습니다.
```bash
> pip install -r requirements_training.txt
```
## 1. 학습 데이터를 준비합니다.
- train/with_mask/ 폴더에는 마스크를 착용한 얼굴 사진이 필요합니다.
- train/without_mask/ 폴더에는 마스크를 착용하지 않은 얼굴 사진이 필요합니다.

기본적으로 리포지토리 내에 이미지가 존재하지만, 사용자가 원하는 사진을 얼마든지 추가해도 무방합니다.<br/>

## 2. 학습 데이터를 전처리합니다.
효과적인 학습을 위해 학습 이미지에서 얼굴만을 추출하여 저장해야 합니다.<br/>
[이미지 전처리 예제](../examples/preprocess_dataset.py)를 통해 사진에서 얼굴을 추출하고 별도 폴더에 저장합니다.
```python3
> python -m examples.preprocess_dataset.py
./dataset/with_mask로부터 학습 데이터 생성 시작
1609it [03:26,  7.79it/s]
./dataset/without_mask로부터 학습 데이터 생성 시작
704it [01:11,  9.85it/s]
```
위 예제를 실행할 경우
- dataset/train/0_mask 에는 마스크 쓴 얼굴 이미지가 저장되며
- dataset/train/1_face 에는 마스크를 쓰지 않은 얼굴 이미지가 저장됩니다.

## 3. 모델을 학습합니다.
[모델 학습 예제(train_model.py)](../examples/train_model.py)에는 전처리된 데이터를 바탕으로 데이터 증강을 수행한 뒤 
간단한 모델을 구현하고 학습합니다. 사용자가 다른 데이터 증강기법이나 모델, 추가적인 작업을 원하는 경우 예제를 수정하여 학습할 수  있습니다.
```python3
> python -m examples.train_model.py
```

## 4. 테스트 세트로 모델 결과를 테스트합니다.
[모델 테스트 예제(test_model.py)](../examples/test_model.py)를 통해 
resource/sample/image 폴더 내의 이미지들을 가져와서 테스트를 수행하고 plot을 표시합니다. 
다른 테스트 세트를 원할 경우 예제 내의 명시된 폴더를 변경하여 수행할 수 있습니다.
```python3
> python -m examples.test_model.py
```
아래는 테스트 결과 예시입니다.<br/>
![test.png](../resource/readme/test.png)