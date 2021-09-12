import sys
import cv2
from cv2 import cv2
import numpy as np
import tensorflow as tf

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('vtest.avi')

if not cap.isOpened():
    print('Camera not found!')
    sys.exit()

# Load network

# net = cv2.dnn.readNet('20210910.model')

# if net.empty():
#     print('Network load failed!')
#     exit()

# Load class names

# classNames = None
# with open('classification_classes_ILSVRC2012.txt', 'rt') as f:
#     classNames = f.read().rstrip('\n').split('\n')

model = tf.keras.models.load_model('./AI_Mask_Detector/20210912.h5')

# #Create probability model 
probability_model = tf.keras.Sequential([model])

width = 128
height = 128

while True:
    ret, frame = cap.read()
	
    if ret:
        #cv2.imshow('frame', frame)

        resizeImg = cv2.resize(frame, (width , height))
        #np_image_data = np.asarray(inp)
        #rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        rgb_tensor = tf.convert_to_tensor(resizeImg, dtype=tf.float32)
        rgb_tensor /= 255.
        rgb_tensor = tf.expand_dims(rgb_tensor , 0)
        
        # inputBlob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 117, 123))
        # probability_model.setInput(inputBlob)
        # prob = probability_model.forward()
 
        
        
        #cv2.imwrite('frame.jpg', frame)
        #cv2.imwrite('resizeImg.jpg', resizeImg)

        #print(rgb_tensor.shape)
        #print(rgb_tensor[0][0])


        
        
        # #Predict label
        predictions = probability_model.predict(rgb_tensor)

        label = 'test'
        if predictions[0][0] > predictions[0][1]:
            label = 'Mask'
        else:
            label = 'No Mask'

        print(predictions[0][0], '   ', predictions[0][1])
            
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)        
        cv2.imshow('frame', frame)

        if cv2.waitKey(30) == 27:
            break
    else:
        print('error')

cap.release()
cv2.destroyAllWindows()	 