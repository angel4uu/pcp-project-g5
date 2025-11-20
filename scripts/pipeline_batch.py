from ultralytics import YOLO
import cv2
import time
import numpy as np
import torch

# Configuración del Batch
BATCH_SIZE = 64

# 0. Definir el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

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

# 3. Inferencia (modelo YOLO) - Acepta una lista de frames
def inferir_batch(model, batch_frames):
    # model() en Ultralytics acepta una lista de numpy arrays
    # verbose=False para no llenar la consola
    results = model(batch_frames, verbose=False, task='detect')
    return results

# 4. Postprocesamiento
def postprocesar_batch(results):
    total_rostros = 0
    # results es una lista de resultados, uno por cada frame del batch
    for result in results:
        rostros_en_frame = 0
        for box in result.boxes:
            # No movemos a CPU aquí para no ralentizar el benchmark puro
            conf = float(box.conf[0].cpu())
            if conf > 0.5:
                rostros_en_frame += 1
        total_rostros += rostros_en_frame
    return total_rostros

# Ejecutar pipeline con Batching
def ejecutar_pipeline_batch(video_path, model_path):
    print(f"Cargando motor TensorRT: {model_path}")
    print(f"Configurado para BATCH SIZE: {BATCH_SIZE}")
    
    model = YOLO(model_path)
    # NO usar model.to(device), el engine ya está en GPU
    
    cap = capturar_video(video_path)

    total_frames = 0
    total_faces = 0
    tiempos = []

    print("Iniciando inferencia por lotes...")

    while True:
        # --- 1. Acumular el Batch ---
        batch_frames = []
        frames_reales = 0
        
        for _ in range(BATCH_SIZE):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = preprocesar(frame)
            batch_frames.append(frame)
            frames_reales += 1
        
        if not batch_frames:
            break
            
        # --- Padding (Relleno) ---
        # Si el batch está incompleto (ej. final del video), TensorRT fallará.
        # Debemos rellenarlo con copias del último frame hasta llegar a 64.
        while len(batch_frames) < BATCH_SIZE:
            batch_frames.append(batch_frames[-1])
            
        start = time.time()
        
        # --- 2. Inferencia (Batch 64) ---
        # Ultralytics se encarga de apilar la lista en un tensor de GPU
        results = inferir_batch(model, batch_frames)
        
        # --- 3. Postprocesamiento ---
        # Solo procesamos los resultados de los frames reales, ignoramos el relleno
        results_reales = results[:frames_reales]
        faces = postprocesar_batch(results_reales)

        # Sincronizar GPU
        torch.cuda.synchronize()

        end = time.time()
        tiempos.append(end - start)

        total_frames += frames_reales
        total_faces += faces

        print(f"Procesados {total_frames} frames...", end='\r')

    cap.release()

    # Métricas de rendimiento
    if tiempos:
        avg_latency_batch = np.mean(tiempos)
        fps = total_frames / sum(tiempos)
        
        # Latencia por frame = Latencia del Batch / Tamaño del Batch
        avg_latency_frame = avg_latency_batch / BATCH_SIZE 

        print(f"\n\n=== RESULTADOS PYTHON (TensorRT Batch {BATCH_SIZE}) ===")
        print(f"Frames procesados: {total_frames}")
        print(f"Rostros detectados: {total_faces}")
        print(f"FPS Promedio: {fps:.2f}")
        print(f"Latencia promedio por BATCH: {avg_latency_batch:.4f} s")
        print(f"Latencia promedio por FRAME: {avg_latency_frame:.4f} s")

# main, ejecución
if __name__ == "__main__":
    video_path = "videos/prueba1.mp4"
    model_path = "models/best.engine" 
    
    if str(device) == "cuda":
        ejecutar_pipeline_batch(video_path, model_path)
    else:
        print("CUDA no está disponible.")