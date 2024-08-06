import cv2
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO('best.pt')  # Model dosyasını uygun olanla değiştirin

# Kamerayı aç
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Genişlik
cap.set(4, 480)  # Yükseklik

while True:
    ret, img = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # Modeli çalıştır ve sonuçları al
    results = model(img)

    # Sonuçları çiz
    annotated_img = results[0].plot()

    # Sonuçları görüntüle
    cv2.imshow('Webcam', annotated_img)

    if cv2.waitKey(1) == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()

# Sonuçları yazdır
print(results)
