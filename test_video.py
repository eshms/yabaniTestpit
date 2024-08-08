import os
import cv2
from ultralytics import YOLO

VIDEO_PATH = 'yabani.mp4'  # Video file name
VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, VIDEO_PATH)
cap = cv2.VideoCapture(video_path)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
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

        model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights','best.pt')

        # Load a model
        model = YOLO(model_path)  # load a custom model

        threshold = 0.5

        while ret:
            results = model(frame)[0]
            found = False

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                if score <= threshold:
                    continue

                label = f"{results.names[int(class_id)].upper()} {score:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                # Print "Found" if class_id is detected ('bikti')
                print(f"Found: {results.names[int(class_id)].upper()}")
                found = True

            if not found:
                print("Not Found")

            out.write(frame)
            cv2.imshow('YOLO Detection', frame)

            # Break the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()
        out.release()
        cv2.destroyAllWindows()
