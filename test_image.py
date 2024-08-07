import os
import cv2
from ultralytics import YOLO

# YOLO modelini yükle
model_path = 'runs/detect/train/weights/last.pt'  # Model dosyasını uygun olanla değiştirin
model = YOLO(model_path)

IMAGE_PATH = 'test1.jpg'  # Resim dosyasının adını uygun olanla değiştirin

IMAGES_DIR = os.path.join('.', 'images')
# pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['./images/test1.jpg', "./images/bitkim.jpg","./images/test2.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk