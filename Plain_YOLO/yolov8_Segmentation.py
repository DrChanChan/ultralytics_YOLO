from ultralytics import YOLO
import cv2
import numpy as np
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Start webcam
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# Load the model
model = YOLO("./model/yolov8n-seg.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    for r in results:
        if r.masks is None:
            continue
        
        masks = r.masks  # Segmentation masks

        for i, mask in enumerate(masks):
            # Convert mask to binary
            mask_data = mask.data.cpu().numpy().astype(np.uint8).squeeze()

            # Overlay mask on image
            color = (0, 255, 0)  # Color for segmentation mask
            mask_color = np.zeros_like(img)
            mask_color[:, :, 1] = mask_data * 255  # Green channel
            
            img = cv2.addWeighted(img, 1, mask_color, 0.5, 0)

            # Class name
            cls = int(r.boxes.cls[i])
            print("Class name -->", classNames[cls])

            # Object details
            bbox = r.boxes.xyxy[i]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            text_color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, text_color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

