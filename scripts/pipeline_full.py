# Pipeline full

import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
import threading
import queue
import cupy as cp

# --- CONFIGURACIÓN ---
BATCH_SIZE = 64
VIDEO_PATH = "videos/prueba1.mp4"
MODEL_PATH = "models/best.engine"  # Debe ser el engine con batch=64

# Para que se vea el efecto del NMS, se debe retornar muchas cajas,
# es decir, aumentar el iou alto en YOLO

# 0. Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if str(device) == "cpu":
    print("ERROR: Se requiere GPU para TensorRT.")
    exit()


# -----------------------
# FUNCIONES DE NMS
# -----------------------
def compute_iou_copy(box, boxes):
    # box: [x1, y1, x2, y2]
    # boxes: [[x1, y1, x2, y2], ...]
    xA = cp.maximum(box[0], boxes[:, 0])
    yA = cp.maximum(box[1], boxes[:, 1])
    xB = cp.maximum(box[2], boxes[:, 2])
    yB = cp.maximum(box[3], boxes[:, 3])

    inter = cp.maximum(0, xB - xA) * cp.maximum(0, yB - yA)
    areaA = (box[2] - box[0]) * (box[3] - box[1])
    areaB = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = areaA + areaB - inter

    # Evitar división por cero
    return inter / (union + 1e-7)


def nms_cupy(boxes, scores, iou_threshold=0.45):
    """
    Aplica Non-Maximum Suppression usando CuPy en GPU.
    boxes: cp.array shape (N, 4)
    scores: cp.array shape (N,)
    """
    # Ordena por puntaje descendente
    idxs = cp.argsort(scores)[::-1]
    keep_indices = []

    while idxs.size > 0:
        # Seleccionar el índice con mayor score actual
        i = idxs[0]
        keep_indices.append(i)

        if idxs.size == 1:
            break
        
        # Calcular IoU del box seleccionado vs. el resto
        ious = compute_iou_copy(boxes[i], boxes[idxs[1:]])
        
        # Mantener solo los que tienen IoU menor al umbral (no se superponen demasiado)
        # idxs[1:] son los candidatos restantes, filtramos esos
        idxs = idxs[1:][ious < iou_threshold]
    
    # Retornamos las cajas filtradas
    return boxes[cp.array(keep_indices)]

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
    # Agnostic NMS=False y conf baja para dejar pasar más cajas 
    # y que tu NMS de CuPy tenga trabajo que hacer.
    return model(batch_frames, verbose=False, task='detect', conf=0.25, iou=0.9)


def postprocesar_batch(results):
    total_rostros = 0

    # Iteramos sobre los resultados (uno por imagen en el batch)
    for result in results:
        # 1. Obtener tensores de PyTorch (están en GPU)
        # xyxy: coordenadas, conf: confianza
        boxes_torch = result.boxes.xyxy
        scores_torch = result.boxes.conf

        # 2. Filtrado inicial por confianza (Vectorizado en PyTorch)
        # Esto es mucho más rápido que el 'if' dentro de un bucle for
        mask = scores_torch > 0.5
        filtered_boxes_torch = boxes_torch[mask]
        filtered_scores_torch = scores_torch[mask]

        if len(filtered_boxes_torch) == 0:
            continue

        # 3. Interoperabilidad PyTorch -> CuPy
        # cp.asarray puede leer tensores de torch directamente gracias a __cuda_array_interface__
        # Esto sucede SIN copiar a CPU.
        boxes_cupy = cp.asarray(filtered_boxes_torch)
        scores_cupy = cp.asarray(filtered_scores_torch)

        # 4. Ejecutar TU función NMS personalizada
        final_boxes = nms_cupy(boxes_cupy, scores_cupy, iou_threshold=0.45)

        # 5. Contar rostros resultantes
        total_rostros += len(final_boxes)

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
