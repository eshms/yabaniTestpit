import cv2
from ultralytics import YOLO

# YOLO modelini yükle
model_path = 'runs/detect/train/weights/best.pt'  # Model dosyasını uygun olanla değiştirin
model = YOLO(model_path)  # load a custom model

cap = cv2.VideoCapture(0)  # Open the default webcam (index 0)
video_path_out = 'webcam_out.mp4'  # Output video file name

cap.set(3, 640)  # Genişlik
cap.set(4, 480)  # Yükseklik

if not cap.isOpened():
    print("Error: Cannot open webcam")
else:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from webcam")
    else:
        H, W, _ = frame.shape
        out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

        threshold = 0.5

        while ret:
            results = model(frame)  # Get predictions
            boxes = results[0].boxes.data.tolist()  # Access results

            # Debug: Print the results to see if the model is returning any predictions
            print(f"Detected {len(boxes)} objects")

            for result in boxes:
                x1, y1, x2, y2, score, class_id = result

                if score > threshold:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, model.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            # Display the frame with detections
            cv2.imshow('YOLO Webcam', frame)
            
            # Write the frame to the output video
            out.write(frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()
        out.release()
        cv2.destroyAllWindows()
