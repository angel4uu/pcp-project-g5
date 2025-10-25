from ultralytics import YOLO
import cv2
import time
import numpy as np

# 1. Lectura de frames
def capturar_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video.")
    return cap

# 2. Preprocesamiento
def preprocesar(frame):
    # Aquí puedes añadir resize, normalización, etc.
    frame_resized = cv2.resize(frame, (640, 480))
    return frame_resized

# 3. Inferencia (modelo YOLO)
def inferir(model, frame):
    results = model(frame, verbose=False)
    return results[0]

# 4. Postprocesamiento
def postprocesar(frame, results):
    rostros_detectados = 0
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf > 0.5:
            rostros_detectados += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame, rostros_detectados

# 5. Visualización
def visualizar(frame):
    cv2.imshow("Detección Secuencial", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        return False
    return True

# Ejecutar pipeline
def ejecutar_pipeline(video_path, model_path):
    model = YOLO(model_path)
    cap = capturar_video(video_path)

    total_frames = 0
    total_faces = 0
    tiempos = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        frame = preprocesar(frame)
        results = inferir(model, frame)
        frame, faces = postprocesar(frame, results)

        end = time.time()
        tiempos.append(end - start)

        total_frames += 1
        total_faces += faces

        if not visualizar(frame):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Métricas de rendimiento
    duracion_total = sum(tiempos)
    fps = total_frames / duracion_total
    latencia = np.mean(tiempos)

    print("\n=== RESULTADOS SECUENCIALES ===")
    print(f"Frames procesados: {total_frames}")
    print(f"Rostros detectados: {total_faces}")
    print(f"FPS promedio: {fps:.2f}")
    print(f"Latencia promedio por frame: {latencia:.4f} s")

# main, ejecución
if __name__ == "__main__":
    video_path = "videos/prueba1.mp4"  # o donde esté el video
    model_path = "../yolov8n-widerface-v2/best.pt"      # o donde esté el modelo
    ejecutar_pipeline(video_path, model_path)