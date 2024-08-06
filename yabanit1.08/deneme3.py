import cv2
from ultralytics import YOLO

model =YOLO('best.pt')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
   ret, img= cap.read()
   results = model(img, stream=True)

   cv2.imshow('Webcam', img)

   if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(results)



