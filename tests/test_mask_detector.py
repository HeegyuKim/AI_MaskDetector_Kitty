import pytest
import cv2
import numpy as np
import tensorflow as tf
from mask_detector import MaskDetector, FacenetDetector


test_image_filename = "resource/sample/image/pexels-gustavo-fring-4127449.jpg"

@pytest.fixture
def faces():
    facenet_detector = FacenetDetector()
    faces, _, _ = facenet_detector.detect_faces_from_file(test_image_filename)
    return faces
    
@pytest.fixture
def mask_detector():
    return MaskDetector()
    


def test_predict(mask_detector, faces):
    """
        얼굴 사진들을 잘 예측했는지 확인한다
    """
    preds = mask_detector.predict(faces)
    
    assert preds.shape == (4, )
    assert all(isinstance(p, np.float32) for p in preds)
    
    # 4개의 얼굴 모두 마스크를 썼다고 인식해야함
    assert all(p > 0.5 for p in preds)
    

def test_predict_one(mask_detector, faces):
    """
        얼굴 사진 1장을 잘 예측했는지 확인한다
    """
    p = mask_detector.predict_one(faces[0])
    
    assert isinstance(p, np.float32)
    assert p > 0.5
    
def test_invalid_format(mask_detector):
    """
        64x64x3 이 아닌 이미지가 입력되면 ValueError 발생
    """
    image = np.zeros((200, 200, 3), dtype=np.int8)
    with pytest.raises(ValueError):
        mask_detector.predict_one(image)