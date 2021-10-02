import pytest
import cv2
import numpy as np
from mask_detector import OpenCVFaceDetector, FacenetDetector


@pytest.fixture
def ocv_detector():
    opencv_model_path = './res10_300x300_ssd_iter_140000_fp16.caffemodel'
    opencv_config_path = './deploy.prototxt'
    return OpenCVFaceDetector(opencv_model_path, opencv_config_path)
    
@pytest.fixture
def facenet_detector():
    return FacenetDetector()
    
@pytest.fixture
def test_image():
    img = cv2.imread("demoImage/pexels-gustavo-fring-4127449.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    

def test_facenet_detect_faces_empty(facenet_detector):
    img = np.zeros((100, 100, 3), np.uint8)
    faces, probs, boxes = facenet_detector.detect_faces(img)
    assert len(faces) == 0
    assert len(probs) == 0
    assert len(boxes) == 0
    
def test_facenet_detect_faces_4_people(facenet_detector, test_image):
    """
        사람 4명이 있는 사진에서 정확하게 얼굴 4개를 검출해내는지 확인합니다.
    """
    faces, probs, boxes = facenet_detector.detect_faces(test_image)
    
    assert len(faces) == 4
    for face in faces:
        assert face.shape == (64, 64, 3)
        assert face.dtype == np.float32
        
    assert len(probs) == 4
    for prob in probs:
        assert isinstance(prob, np.float32)
        
    assert len(boxes) == 4
    assert boxes.shape == (4, 4)
    