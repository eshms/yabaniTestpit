import os
import cv2
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO('runs/detect/train/weights/last.pt')  # Model dosyasını uygun olanla değiştirin

VIDEO_PATH = 'lettuce_y.mp4'  # Video dosyasının adını uygun olanla değiştirin

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, VIDEO_PATH)
cap = cv2.VideoCapture(video_path)
cap.set(3, 640)  # Genişlik
cap.set(4, 480)  # Yükseklik
video_path_out = '{}_out.mp4'.format(os.path.splitext(video_path)[0])

if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
else:
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Cannot read the first frame of the video file {video_path}")
    else:
        H, W, _ = frame.shape
        out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

        model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

        # Load a model
        model = YOLO(model_path)  # load a custom model

        threshold = 0.5

        while ret:
            results = model(frame)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                if score > threshold:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            out.write(frame)
            ret, frame = cap.read()

        cap.release()
        out.release()
        cv2.destroyAllWindows()
