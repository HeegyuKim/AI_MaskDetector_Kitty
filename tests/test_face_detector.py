import pytest
import cv2
import numpy as np
from mask_detector import FacenetDetector


test_image_filename = "resource/sample/image/pexels-gustavo-fring-4127449.jpg"


@pytest.fixture
def facenet_detector():
    return FacenetDetector()


@pytest.fixture
def test_image():
    img = cv2.imread(test_image_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def test_facenet_detect_faces_empty(facenet_detector):
    """
    비어있는 이미지에서는 아무 얼굴도 검출하지 못한다.
    """
    img = np.zeros((100, 100, 3), np.uint8)
    faces, probs, boxes = facenet_detector.detect_faces(img)
    assert len(faces) == 0
    assert len(probs) == 0
    assert len(boxes) == 0


def assert_4_people(faces, probs, boxes):
    assert len(faces) == 4
    for face in faces:
        assert face.shape == (64, 64, 3)
        assert face.dtype == np.float32

    assert len(probs) == 4
    for prob in probs:
        assert isinstance(prob, np.float32)

    assert len(boxes) == 4
    assert boxes.shape == (4, 4)


def test_facenet_detect_faces_4_people(facenet_detector, test_image):
    """
    FacenetDetector가 사람 4명이 있는 사진에서 정확하게 얼굴 4개를 검출한다.
    """
    faces, probs, boxes = facenet_detector.detect_faces(test_image)
    assert_4_people(faces, probs, boxes)


def test_facenet_detect_faces_4_people(facenet_detector):
    """
    detect_faces_from_file의 결과물이 detect_faces 처럼 얼굴 4개를 검출합니다.
    """
    faces, probs, boxes = facenet_detector.detect_faces_from_file(test_image_filename)
    assert_4_people(faces, probs, boxes)
