import cv2
import time
import numpy as np
import os
from ultralytics import YOLO
from multiprocessing import Process, Queue
import torch

# 0. Definir el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
print(
    f"Dispositivo CUDA disponible: {torch.cuda.get_device_name(0)}"
    if torch.cuda.is_available()
    else "CUDA no disponible"
)


# 1. Lectura de frames
def capturar_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video.")
    return cap


# 2. Preprocesamiento
def preprocesar(frame):
    frame_resized = cv2.resize(frame, (640, 480))
    return frame_resized


# Worker de captura y preprocesamiento
def worker_preparacion_cpu(video_path, output_queue):
    cap = capturar_video(video_path)
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
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
        conf = float(box.conf[0].cpu())
        if conf > 0.5:
            rostros_detectados += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame, rostros_detectados


# 5. Visualización
def visualizar(frame):
    cv2.imshow("Deteccion Paralela (CPU)", frame)
    return cv2.waitKey(1) & 0xFF != 27


# Ejecutar pipeline
def ejecutar_pipeline(video_path, model_path):
    # Cargar modelo YOLO y mover a GPU
    model = YOLO(model_path)
    model.to(device)

    # Crear cola para frames preprocesados
    preprocessed_queue = Queue(maxsize=5)

    # Iniciar proceso de captura y preprocesamiento
    prepro_process = Process(
        target=worker_preparacion_cpu,
        args=(video_path, preprocessed_queue),
    )
    prepro_process.daemon = True
    prepro_process.start()

    # Inicializar contadores y lista para tiempos
    total_frames = 0
    total_faces = 0
    tiempos_infer_post = []

    # Bucle principal del consumidor
    while True:
        # Obtener frame preprocesado de la cola
        frame_preprocesado = preprocessed_queue.get()
        if frame_preprocesado is None:
            break

        # Medir tiempo de inferencia + postprocesamiento
        start = time.time()

        # Inferencia
        results = inferir(model, frame_preprocesado)

        # Postprocesamiento
        frame_postprocesado, faces = postprocesar(frame_preprocesado, results)

        # Asegurar sincronización de GPU
        torch.cuda.synchronize()

        end = time.time()

        # Registrar tiempo de inferencia + postprocesamiento
        tiempos_infer_post.append(end - start)

        # Actualizar contadores
        total_frames += 1
        total_faces += faces

        # Visualización
        if not visualizar(frame_postprocesado):
            break

    # Limpiar y cerrar
    prepro_process.join(timeout=1)
    cv2.destroyAllWindows()

    # Mostrar resultados
    if tiempos_infer_post:
        # Mide el rendimiento del Consumidor
        duracion_total = sum(tiempos_infer_post)
        fps_consumidor = total_frames / duracion_total
        latencia_consumidor = np.mean(tiempos_infer_post)

        print("\n=== RESULTADOS PARALELOS (HU-02: GPU + CPU-Prepro) ===")
        print(f"Frames procesados: {total_frames}")
        print(f"Rostros detectados: {total_faces}")
        print(f"FPS (Consumidor): {fps_consumidor:.2f}")
        print(f"Latencia promedio (Infer + Postpro): {latencia_consumidor:.4f} s")
    else:
        print("No se procesaron frames.")


# main, ejecución
if __name__ == "__main__":
    video_path = "videos/prueba1.mp4"  # o donde esté el video
    model_path = "../yolov8n-widerface-v2/best.pt"  # o donde esté el modelo

    # Comprueba si hay CUDA y solo entonces ejecuta el pipeline
    if str(device) == "cuda":
        ejecutar_pipeline(video_path, model_path)
    else:
        print("CUDA no está disponible. Ejecución cancelada.")
