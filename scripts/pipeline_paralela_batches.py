"""
Este script implementa un pipeline paralelo del tipo Productor-Consumidor
para maximizar el throughput de la inferencia de rostros en un video.

Técnicas de Paralelismo Aplicadas:
1.  Multiprocesamiento (multiprocessing.Process):
    - Un proceso "Productor" (worker_preparacion_cpu) se dedica
      exclusivamente a leer el video y preprocesar frames (resize) en un
      núcleo de CPU.
    - El proceso "Consumidor" (proceso principal) recibe los frames listos
      para inferencia a través de una Cola (multiprocessing.Queue).
2.  Paralelismo CPU/GPU (Overlap):
    - El Consumidor utiliza la GPU para ejecutar la inferencia (model()).
    - Mientras la GPU está ocupada procesando el "Frame A", la CPU ya está
      preparando el "Frame B" en paralelo, ocultando la latencia de
      preprocesamiento.
3.  Batching:
    - El Consumidor agrupa los frames preprocesados en lotes (batches)
      antes de enviarlos a la GPU, mejorando drásticamente el
      throughput de la inferencia.
"""

import cv2
import time
import numpy as np
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

# Tamaño del lote para inferencia
BATCH_SIZE = 4


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


# Worker de captura y preprocesamiento (Paralelo en CPU)
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


# 3. Inferencia (Paralelo en GPU)
def inferir(model, batch_de_frames):
    # Procesa la lista completa de frames como un lote
    results_list = model(batch_de_frames, verbose=False, stream=False)
    return results_list


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
    cv2.imshow("Deteccion Paralela (GPU)", frame)
    return cv2.waitKey(1) & 0xFF != 27


# Ejecutar pipeline
def ejecutar_pipeline(video_path, model_path):
    # Cargar modelo YOLO y mover a GPU
    model = YOLO(model_path)
    model.to(device)

    # Crear cola para frames preprocesados
    preprocessed_queue = Queue(
        maxsize=BATCH_SIZE * 2
    )  # Tamaño de cola mayor que el lote

    # Iniciar proceso de captura y preprocesamiento
    prepro_process = Process(
        target=worker_preparacion_cpu,
        args=(video_path, preprocessed_queue),
    )
    prepro_process.daemon = True
    prepro_process.start()

    # Inicializar contadores y listas para tiempos
    total_frames = 0
    total_faces = 0
    tiempos_infer_post = []
    tiempos_lote = []  # Tiempo por lote

    # Bandera para manejar el fin del video
    video_terminado = False

    # Bucle principal del consumidor
    while not video_terminado:
        # Acumular un lote de frames desde la cola
        batch_preprocesado = []
        for _ in range(BATCH_SIZE):
            frame = preprocessed_queue.get()
            if frame is None:
                video_terminado = True  # El productor terminó
                break
            batch_preprocesado.append(frame)

        # Si el lote está vacío (porque el video terminó), salir
        if not batch_preprocesado:
            break

        # Guardar el número real de frames en este lote (puede ser < BATCH_SIZE al final)
        frames_en_lote_actual = len(batch_preprocesado)

        # Medir tiempo de inferencia + postprocesamiento del lote
        start = time.time()

        # Inferencia
        results_list = inferir(model, batch_preprocesado)

        # Crear listas para guardar los resultados del postprocesamiento
        frames_para_visualizar = []
        rostros_en_lote = 0

        # Postprocesamiento
        for i in range(frames_en_lote_actual):
            frame_original = batch_preprocesado[i]
            results_individual = results_list[i]

            frame_post, faces = postprocesar(frame_original, results_individual)
            rostros_en_lote += faces
            frames_para_visualizar.append(frame_post)

        # Asegurar sincronización de GPU
        torch.cuda.synchronize()
        end = time.time()

        # Registrar tiempo del LOTE
        tiempos_lote.append(end - start)

        # Visualización
        for frame_a_visualizar in frames_para_visualizar:
            if not visualizar(frame_a_visualizar):
                video_terminado = True
                break

        # Actualizar contadores
        total_frames += frames_en_lote_actual
        total_faces += rostros_en_lote

    # Limpiar y cerrar
    prepro_process.join(timeout=1)
    cv2.destroyAllWindows()

    # Mostrar resultados
    if tiempos_lote:
        duracion_total = sum(tiempos_lote)
        fps_total = total_frames / duracion_total
        latencia_lote_promedio = np.mean(tiempos_lote)
        latencia_frame_promedio = duracion_total / total_frames

        print("\n=== RESULTADOS PARALELOS CON BATCHING ===")
        print(f"Número de batches: {BATCH_SIZE}")
        print(f"Frames procesados: {total_frames}")
        print(f"Rostros detectados: {total_faces}")
        print(f"Tiempo total: {duracion_total:.2f} s")
        print(f"FPS promedio: {fps_total:.2f}")
        print(f"Latencia promedio por lote: {latencia_lote_promedio:.4f} s")
        print(f"Latencia promedio por frame: {latencia_frame_promedio:.4f} s")
    else:
        print("No se procesaron frames.")


# main, ejecución
if __name__ == "__main__":
    video_path = "./videos/prueba2.mp4"  # o donde esté el video
    model_path = "../yolov8n-widerface-v2/best.pt"  # o donde esté el modelo

    # Comprueba si hay CUDA y solo entonces ejecuta el pipeline
    if str(device) == "cuda":
        ejecutar_pipeline(video_path, model_path)
    else:
        print("CUDA no está disponible. Ejecución cancelada.")
