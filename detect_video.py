import cv2
import tensorflow as tf
from tensorflow import keras
from facenet_pytorch import MTCNN

from model.hkmodel import SimpleCNNModel
import numpy as np
from PIL import Image, ImageDraw


input_file = './demoVideo/test3.mp4'
output_file = './movie_output.mp4'

in_cap = cv2.VideoCapture(input_file)

if not in_cap.isOpened(): 
    print(f"파일을 열 수 없습니다: {input_file}")
    exit(0)
    
width  = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
fps = int(in_cap.get(cv2.CAP_PROP_FPS))

print('width, height, fps :', width, height, fps)


out_cap = cv2.VideoWriter(output_file, 0x7634706d, int(fps), (int(width), int(height)))
mtcnn = MTCNN(image_size=224, margin=0, keep_all=True, post_process=False, device='cpu')

model = SimpleCNNModel()
# model.compile(optimizer='adam', loss='sparse_c1ategorical_crossentropy', metrics=['accuracy'])
model.build(input_shape = (None, 224, 224, 3))
model.load_weights("model-best.h5")


while True:
    ret, frame = in_cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    boxes, probs = mtcnn.detect(Image.fromarray(frame), landmarks=False)
    
    
    if boxes is not None:
        try:
            faces = [frame[int(y):int(h), int(x):int(w)] for x, y, w, h in boxes]
            faces = [cv2.resize(face, (224, 224)) for face in faces]
            faces = np.stack(faces, axis=0)
            masks_prob = model.predict(faces)
            masks = [x[1] > x[0] for x in masks_prob]
            
            for p, mask, box, fp in zip(masks_prob, masks, boxes, probs):
                color = (0, 255, 0) if mask else (255, 0, 0)
                
                x1, y1 = int(box[0]), int(box[1])
                x2, y2 = int(box[2]), int(box[3])
                
                label = 'Mask ' + str(p[1]) if mask else 'Face ' + str(p[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color)
                cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"Face Prob: {fp}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                
                # cv2.putText(frame_draw, label, (box[0], box[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        except Exception as e:
            print(str(e))
        
    out_cap.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
in_cap.release()
out_cap.release()