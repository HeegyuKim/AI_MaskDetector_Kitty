# 동영상 파일에서 사람 얼굴을 찾고 마스크 착용 여부에 따라 표시하여
# 그 결과물을 다른 동영상 파일로 저장하는 예제입니다.
# 아래 명령어를 통해 직접 실행해보세요
# python -m examples.detect_video_masked_face.py
import cv2
from mask_detector import MaskDetector, OpenCVFaceDetector, MaskedFaceDrawer


input_file = "./resource/sample/video/pexels-steven-hause-5827569.mp4"
output_file = "./detected-pexels-steven-hause-5827569.mp4"


# OpenCV를 이용하여 동영상에서 프레임을 읽어오겠습니다.
in_cap = cv2.VideoCapture(input_file)
if not in_cap.isOpened():
    print(f"파일을 열 수 없습니다: {input_file}")
    exit(0)

# 동영상 정보를 받아옵니다(가로, 세로 크기 및 초당 프레임 수)
width = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(in_cap.get(cv2.CAP_PROP_FPS))

print("width, height, fps :", width, height, fps)

# 읽어오는 영상과 동일한 크기와 fps로 저장할 mp4 영상을 만듭니다.
out_cap = cv2.VideoWriter(output_file, 0x7634706D, fps, (width, height))

mask_detector = MaskDetector()
face_detector = FacenetDetector()
mask_drawer = MaskedFaceDrawer(mask_detector, face_detector)

# 영상의 끝까지 프레임을 읽어옵니다.
while True:
    ret, frame = in_cap.read()
    # 더 이상 읽어올 게 없다면 끝
    if not ret:
        break

    # 영상의 프레임을 받아 rgb로 바꾼 후 MaskedFaceDrawer를 이용하여
    # 얼굴에 마스크 착용 여부를 표시합니다.
    # 표시된 이미지는 다시 결과 동영상에 저장합니다.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask_drawer.rectangle_faces(image)
    out_cap.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

in_cap.release()
out_cap.release()
