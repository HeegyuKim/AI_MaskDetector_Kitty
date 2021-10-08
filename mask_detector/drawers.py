import cv2


class MaskedFaceDrawer:
    
    def __init__(self, mask_detector, face_detector):
        self.mask_detector = mask_detector
        self.face_detector = face_detector
        
    def rectangle_faces(
        self, 
        image, 
        mask_color=(0,255,0), 
        no_mask_color=(255,0,0), 
        mask_threshold=0.5,
        draw_text=True
        ):
        
        faces, confidences, boxes = self.face_detector.detect_faces(image)
            
        if len(faces) == 0:
            return
        
        mask_probs = self.mask_detector.predict(faces)
        
        
        for conf, mask_prob, box in zip(confidences, mask_probs, boxes):
            x1, y1, x2, y2 = box
            color = mask_color if mask_prob > mask_threshold else no_mask_color
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color)
            
            if draw_text:
                cv2.putText(image, f"Mask Probability: {mask_prob:.2f}", (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                cv2.putText(image, f"Face Confidence: {conf:.2f}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        