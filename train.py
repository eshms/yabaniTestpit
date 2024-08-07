from ultralytics import YOLO

model = YOLO('yolov8n.yaml')


results = model.train(data='/Users/osman/Documents/GitHub/yabanitestpit2/config.yaml', epochs=5, device='mps',batch=16, workers=8)