#Pipeline full

import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
import threading
import queue

# --- CONFIGURACIÓN ---
BATCH_SIZE = 64
VIDEO_PATH = "videos/prueba1.mp4"
MODEL_PATH = "models/best.engine" # Debe ser el engine con batch=64

# 0. Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if str(device) == "cpu":
    print("ERROR: Se requiere GPU para TensorRT.")
    exit()

# -----------------------
# TAREA 1: EL WORKER (Productor)
# -----------------------
def preprocesar(frame):
    return cv2.resize(frame, (640, 480))

def worker_productor_hilos(video_path, cola_lotes, batch_size):
    print(f"[Hilo Productor] Iniciando lectura...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cola_lotes.put(None)
        return

    while True:
        batch_frames = []
        frames_reales = 0

        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV libera el GIL aquí
            frame = preprocesar(frame)
            batch_frames.append(frame)
            frames_reales += 1

        if not batch_frames:
            cola_lotes.put(None)
            break

        # Padding (Relleno)
        while len(batch_frames) < batch_size:
            batch_frames.append(batch_frames[-1])

        # Enviar referencia a la lista (sin copia de memoria)
        cola_lotes.put((batch_frames, frames_reales))

    cap.release()
    print(f"[Hilo Productor] Fin del video.")

# -----------------------
# TAREA 2: CONSUMIDOR (Main)
# -----------------------

def inferir_batch(model, batch_frames):
    return model(batch_frames, verbose=False, task='detect')

def postprocesar_batch(results):
    total_rostros = 0
    for result in results:
        rostros_en_frame = 0
        for box in result.boxes:
            if float(box.conf[0].cpu()) > 0.5:
                rostros_en_frame += 1
        total_rostros += rostros_en_frame
    return total_rostros

def ejecutar_pipeline_threading():
    print(f"Cargando motor TensorRT: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    queue_lotes = queue.Queue(maxsize=4)

    t = threading.Thread(target=worker_productor_hilos, args=(VIDEO_PATH, queue_lotes, BATCH_SIZE))
    t.start()

    total_frames = 0
    total_faces = 0
    tiempos_gpu = []

    print("Iniciando consumo de lotes (GPU)...")

    # --- CRONÓMETRO GLOBAL (Para FPS Reales) ---
    pipeline_start = time.time()

    while True:
        item = queue_lotes.get()

        if item is None:
            break

        batch_frames, frames_reales = item

        # Medir solo el tiempo de GPU (Latencia de Inferencia)
        start_gpu = time.time()

        results = inferir_batch(model, batch_frames)
        faces = postprocesar_batch(results[:frames_reales])

        torch.cuda.synchronize()
        end_gpu = time.time()

        tiempos_gpu.append(end_gpu - start_gpu)
        total_frames += frames_reales
        total_faces += faces

        print(f"Procesados {total_frames} frames...", end='\r')

    pipeline_end = time.time()
    t.join()

    if tiempos_gpu:
        # 1. FPS Reales (Del sistema completo)
        tiempo_total = pipeline_end - pipeline_start
        fps_reales = total_frames / tiempo_total

        # 2. Latencia GPU (Solo inferencia)
        latencia_gpu_avg = np.mean(tiempos_gpu)

        print(f"\n\n=== RESULTADOS THREADING (Python Hilos + Batch {BATCH_SIZE}) ===")
        print(f"Frames procesados: {total_frames}")
        print(f"Rostros detectados: {total_faces}")
        print("------------------------------------------")
        print(f"Tiempo Total de Ejecución: {tiempo_total:.2f} s")
        print(f"FPS REALES (Sistema): {fps_reales:.2f}")
        print("------------------------------------------")
        print(f"Latencia GPU por Batch: {latencia_gpu_avg:.4f} s")

if __name__ == "__main__":
    ejecutar_pipeline_threading()