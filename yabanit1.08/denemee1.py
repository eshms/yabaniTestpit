from ultralytics import YOLO
import cv2
import math


# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("best.pt")

# object classes
classNames = ["bitkim.jpg"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    results = model(img)

    # coordinates
    for result in results:
        boxes = result.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0].astype(int)

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = (x1, y1 - 10)  # Adjust text position
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
