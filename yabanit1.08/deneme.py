from ultralytics import YOLO
from PIL import Image

model =YOLO('best.pt')

#görseli tanıma
#im1= Image.open("bitkim.jpg")
#sonuc= model.predict(source=im1, save=True) #save resmi kaydeder


#webcam ile tanıma
sonuc= model.predict(source="0") #0 webcam için


#mp4
#sonuc= model.predict(source="yabani.mp4", show=True) #mp4 için


