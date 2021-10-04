## 1. 학습 데이터를 추가합니다.
train/with_mask/ 폴더에는 마스크를 착용한 얼굴 사진을<br/>
train/without_mask/ 폴더에는 마스크를 착용하지 않은 얼굴 사진을 추가합니다.

## 2. 학습 데이터를 전처리합니다.
효과적인 학습을 위해 학습 이미지에서 얼굴만을 추출하여 저장합니다.
- dataset/train/0_mask 에는 마스크 쓴 얼굴 이미지가 저장되며
- dataset/train/1_face 에는 마스크를 쓰지 않은 얼굴 이미지가 저장됩니다.
```python3
> python -m examples.preprocess_dataset.py
./dataset/with_mask로부터 학습 데이터 생성 시작
1609it [03:26,  7.79it/s]
./dataset/without_mask로부터 학습 데이터 생성 시작
704it [01:11,  9.85it/s]
```
## 3. 모델을 학습합니다.
```python3

```

## 4. 테스트 세트로 모델 결과를 검증합니다.
```python3
> python3 -m examples.test_model.py
```

테스트 결과가 test.png 파일로 저장됩니다.<br/>
![test.png](resource/readme/test.png)