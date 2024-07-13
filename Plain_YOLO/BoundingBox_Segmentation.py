from ultralytics import YOLO
import cv2
import math
import numpy as np
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)

#image 
#img_path = cv2.imread("../streetandpo.jpg")
#img = cv2.resize(img_path, (640,640))

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model_seg = YOLO("../streetn.pt")
model_det = YOLO("./pothole.pt")

classNames = ["Crosswalk", "Curb-Down", "Curb-Up", "Floor", "Ground", "Hole", "Road", "Sidewalk", "Stairs-Down", "Stairs-Up"]
classNames2 = ["pothole"]

class_colors = {
    "Crosswalk": (255, 0, 0),      # Red
    "Curb-Down": (0, 255, 0),      # Green
    "Curb-Up": (0, 0, 255),        # Blue
    "Floor": (255, 255, 0),        # Cyan
    "Ground": (255, 0, 255),       # Magenta
    "Hole": (0, 255, 255),         # Yellow
    "Road": (128, 128, 0),         # Olive
    "Sidewalk": (128, 0, 128),     # Purple
    "Stairs-Down": (0, 128, 128),  # Teal
    "Stairs-Up": (128, 0, 0)       # Maroon
}

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.flip(img, 0)
    
    results_seg = model_seg(img, stream=True)
    results_det = model_det(img, stream=True)

    for r in results_seg:
        masks = r.masks  # 세그멘테이션 마스크
        
        if masks is not None:
            for i, mask in enumerate(masks):
                # 마스크를 바이너리 형태로 변환
                mask_data = mask.data.cpu().numpy().astype(np.uint8).squeeze()

                class_name = classNames[i]
                if class_name in class_colors:
                    color = class_colors[class_name]
                else:
                    color = (0,0,0)
                    
                # 마스크를 이미지에 오버레이
                mask_color = np.zeros_like(img)
                mask_color[:, :, 0] = mask_data * color[0]  # Blie channel
                mask_color[:, :, 1] = mask_data * color[1]  # Green channel
                mask_color[:, :, 2] = mask_data * color[2]  # Red channel
                img = cv2.addWeighted(img, 1, mask_color, 0.2, 0)
            
                
                # class name
                cls = int(r.boxes.cls[i])
                             
                # Object details
                bbox = r.boxes.xyxy[i]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                org = (x1, y1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                text_color = (0, 0, 255)
                thickness = 1
                cv2.putText(img, classNames[cls], org, font, fontScale, text_color, thickness)
                


    for r in results_det:
        boxes = r.boxes
        
        if boxes is not None:
            for i, box in enumerate(boxes):
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # confidence
                #confidence = math.ceil((box.conf[i] * 100)) / 100
                #print("Confidence (detection) --->", confidence)

                # class name
                cls = int(box.cls[0])
                #print("Class name (detection) -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 1

                cv2.putText(img, classNames2[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()