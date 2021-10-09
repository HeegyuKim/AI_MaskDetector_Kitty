import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

model = "./AI_Mask_Detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config = "./AI_Mask_Detector/deploy.prototxt"
# model = './AI_Mask_Detector/opencv_face_detector_uint8.pb'
# config = './AI_Mask_Detector/opencv_face_detector.pbtxt'

mask_model = tf.keras.models.load_model("./AI_Mask_Detector/model.h5")
probability_model = tf.keras.Sequential([mask_model])
width = 300
height = 300

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open failed!")
    exit()

net = cv2.dnn.readNet(model, config)

if net.empty():
    print("Net open failed!")
    exit()

while True:
    ret, frame = cap.read()

    if ret:
        blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
        net.setInput(blob)
        detect = net.forward()

        detect = detect[0, 0, :, :]
        (h, w) = frame.shape[:2]

        # print('--------------------------')
        for i in range(detect.shape[0]):
            confidence = detect[i, 2]
            if confidence < 0.5:
                break

            x1 = int(detect[i, 3] * w)
            y1 = int(detect[i, 4] * h)
            x2 = int(detect[i, 5] * w)
            y2 = int(detect[i, 6] * h)

            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

            margin = 0
            face = frame[y1 - margin : y2 + margin, x1 - margin : x2 + margin]

            resize = cv2.resize(face, (width, height))

            # print(x1, y1, x2, y2, width, height)
            cv2.imshow("frame1", resize)

            # np_image_data = np.asarray(inp)
            rgb_tensor = tf.convert_to_tensor(resize, dtype=tf.float32)
            rgb_tensor /= 255.0
            rgb_tensor = tf.expand_dims(rgb_tensor, 0)

            # 예측
            predictions = probability_model.predict(rgb_tensor)

            # 화면 레이블
            label = "test"
            if predictions[0][0] > predictions[0][1]:
                label = "Mask " + str(predictions[0][0])
            else:
                label = "No Mask " + str(predictions[0][1])

            # print(predictions[0][0], '   ', predictions[0][1])

            cv2.putText(
                frame,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("frame", frame)

            # frame[0:width, 0:height] = face

            label = "Face: %4.3f" % confidence
            cv2.putText(
                frame,
                label,
                (x1, y1 - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break

    else:
        print("error")

cap.release()
cv2.destroyAllWindows()
