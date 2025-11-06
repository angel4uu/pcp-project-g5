import cv2
import time
import numpy as np
import os
from ultralytics import YOLO
from multiprocessing import Process, Queue

# Tarea 1: Aislar la función de preprocesamiento
def preprocesar(frame):
    frame_resized = cv2.resize(frame, (640, 480))
    return frame_resized

# Tarea 2: Implementar paralelismo - Uso de workers
def worker_preparacion_cpu(video_path, output_queue):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Worker (PID: {os.getpid()}) no pudo abrir el video.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            output_queue.put(None)
            break
        processed_frame = preprocesar(frame)
        output_queue.put(processed_frame)
    cap.release()

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
    cv2.imshow("Deteccion Paralela (CPU)", frame)
    return cv2.waitKey(1) & 0xFF != 27

# Tarea 3: Integración del módulo paralelo
if __name__ == "__main__":
    print("Usando dispositivo: cpu")

    script_dir = os.path.dirname(__file__)
    video_path = os.path.join(script_dir, '.', 'videos', 'prueba3.mp4')
    model_path = os.path.join(script_dir, '..', 'yolov8n-widerface-v2', 'best.pt')
    model_path = "../yolov8n-widerface-v2/best.pt"

    # Cargamos el modelo en el proceso principal (usará CPU)
    model = YOLO(model_path)

    preprocessed_queue = Queue(maxsize=4)
    prepro_process = Process(target=worker_preparacion_cpu, args=(video_path, preprocessed_queue))
    prepro_process.daemon = True
    prepro_process.start()
    
    total_frames = 0
    total_faces = 0
    tiempos_infer_post = []

    while True:
        frame_preprocesado = preprocessed_queue.get()
        if frame_preprocesado is None:
            break

        start = time.time()
        
        results = inferir(model, frame_preprocesado)
        frame_postprocesado, faces = postprocesar(frame_preprocesado, results)
        
        end = time.time()
        tiempos_infer_post.append(end - start)
        
        total_frames += 1
        total_faces += faces

        if not visualizar(frame_postprocesado):
            break

    prepro_process.join(timeout=1)
    cv2.destroyAllWindows()

    if tiempos_infer_post:
        duracion_total = sum(tiempos_infer_post)
        fps = total_frames / duracion_total
        latencia = np.mean(tiempos_infer_post)

        print("\n=== RESULTADOS PARALELOS (usando CPU) ===")
        print(f"Frames procesados: {total_frames}")
        print(f"Rostros detectados: {total_faces}")
        print(f"FPS promedio: {fps:.2f}")
        print(f"Latencia promedio (Inferencia + Postpro): {latencia:.4f} s")