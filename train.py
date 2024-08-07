from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

# change device to '0' for CPU ,'1' for GPU and 'mps' for M1 chips
results = model.train(data='/Users/osman/Documents/GitHub/yabanitestpit2/config.yaml', epochs=5, device='mps',batch=16, workers=8)