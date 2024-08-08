from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

# change device to '0' for CPU ,'1' for GPU and 'mps' for M1 chips
# chnage config file to your own config file path(use abosulte path)
results = model.train(data='data.yaml', epochs=100,imgsz=640,device='mps')