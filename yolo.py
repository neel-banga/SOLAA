import cv2
import numpy as np
from ultralytics import YOLO
import time
import face_det
import audio
import os

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')

said = []

def yolo():
    num = 0
    start_time = time.time()
    while time.time() - start_time < 20:
        num += 1
        ret, frame = cap.read()
        # Pass the frame through the YOLO model for prediction
        results = model.predict(frame)
        if results and len(results[0].boxes) > 0:
            result = results[0]
            box = result.boxes[0]
            for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)
                print("Object type:", class_id)
                print("Coordinates:", cords)
                print("Probability:", conf)
                print("---")
                image = cv2.rectangle(frame, (cords[0],cords[1]), (cords[2],cords[3]), (255,0,0), 2)
                cv2.putText(image, text= str(class_id), org=(cords[0],cords[1]),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0),
                    thickness=2, lineType=cv2.LINE_AA)
                cv2.imshow('yolo', image)

                if class_id.lower() == 'person':
                    path = ''
                    num +=1
                    cv2.imwrite(f"image{num}.jpg", frame)
                    name = face_det.check(f"image{num}.jpg")
                    os.remove(f"image{num}.jpg")
                    if name not in said:
                        audio.say(name)
                        said.append(name)
                else:
                    if name not in said:
                        audio.say(class_id)
                        said.append(class_id)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
