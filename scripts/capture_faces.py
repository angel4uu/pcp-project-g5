from ultralytics import YOLO
import cv2
import os
import imutils
import argparse

# 🧾 Argument parser
parser = argparse.ArgumentParser(description="Captura de rostros con YOLOv8")
parser.add_argument('--person', type=str, required=True, help='Nombre de la persona')
parser.add_argument('--count', type=int, default=300, help='Número de imágenes a capturar')
args = parser.parse_args()

# 📁 Nombre y ruta
personName = args.person
maxCount = args.count
dataPath = '../lbph_data/faces'
personPath = os.path.join(dataPath, personName)

if not os.path.exists(personPath):
    print('📁 Carpeta creada:', personPath)
    os.makedirs(personPath)

# 🔍 Carga del modelo YOLOv8
model = YOLO('../model.pt')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    auxFrame = frame.copy()

    results = model(frame, verbose=False)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        rostro = auxFrame[y1:y2, x1:x2]
        if rostro.size == 0:
            continue  # Evita errores si el recorte es inválido

        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'{personPath}/rostro_{count}.jpg', rostro)
        count += 1

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27 or count >= maxCount:
        break

cap.release()
cv2.destroyAllWindows()

