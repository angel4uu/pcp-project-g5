"""
Este script implementa un pipeline de inferencia de detección de rostros estrictamente
secuencial que utiliza la aceleración de la Unidad de Procesamiento Gráfico (GPU)
a través de CUDA y PyTorch.

Características del Flujo:
1.  Aceleración de Cálculo (GPU Paralelo):
    - El modelo de YOLO se mueve a la memoria de la GPU (model.to('cuda')).
    - La etapa de inferencia (model()) se ejecuta en paralelo en los miles de núcleos
      de la GPU, lo que reduce drásticamente el tiempo de cálculo.
2.  Flujo Secuencial:
    - Las etapas del pipeline (Captura, Preprocesamiento, Inferencia, Postprocesamiento
      y Visualización) se ejecutan de forma secuencial. La CPU debe esperar
      a que cada etapa termine antes de pasar a la siguiente.
"""

from ultralytics import YOLO
import cv2
import time
import numpy as np
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


# 3. Inferencia (modelo YOLO)
def inferir(model, frame):
    results = model(frame, verbose=False)
    return results[0]


# 4. Postprocesamiento
def postprocesar(frame, results):
    rostros_detectados = 0
    for box in results.boxes:
        # Mover tensores a la CPU con .cpu()
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
        conf = float(box.conf[0].cpu())
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
    # Mover el modelo a la GPU
    model.to(device)

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

        # Sincronizar para asegurar que todo el trabajo
        # de la GPU (inferir + postpro .cpu()) se mida.
        torch.cuda.synchronize()

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

    print("\n=== RESULTADOS SECUENCIALES (CUDA)===")
    print(f"Frames procesados: {total_frames}")
    print(f"Rostros detectados: {total_faces}")
    print(f"FPS promedio: {fps:.2f}")
    print(f"Latencia promedio por frame: {latencia:.4f} s")


# main, ejecución
if __name__ == "__main__":
    video_path = "videos/prueba1.mp4"  # o donde esté el video
    model_path = "../yolov8n-widerface-v2/best.pt"  # o donde esté el modelo
    # Esto es para asegurar que se use CUDA.
    if str(device) == "cuda":
        ejecutar_pipeline(video_path, model_path)
    else:
        print("CUDA no está disponible. Ejecución cancelada.")
