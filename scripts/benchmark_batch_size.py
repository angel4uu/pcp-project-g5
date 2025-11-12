"""
Script de BENCHMARKING (Evaluación de Rendimiento) para encontrar el
tamaño de lote (BATCH_SIZE) óptimo y ESTABLE.

Técnica:
1.  Define un NÚMERO DE EJECUCIONES (ej. 3).
2.  Define una LISTA DE LOTES a probar (ej. [1, 2, 4, 8, 16, 32, 64]).
3.  Ejecuta la prueba completa N veces, con una pausa entre ellas
    para estabilización térmica.
4.  Calcula el PROMEDIO, MÍNIMO, MÁXIMO y DESVIACIÓN ESTÁNDAR (StdDev)
    para cada tamaño de lote.
5.  Recomienda el lote más ESTABLE (bajo StdDev) que siga estando
    dentro del 95% del rendimiento pico.

"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from multiprocessing import Process, Queue
import torch

# 0. Definir el dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
if torch.cuda.is_available():
    print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")


# --- Funciones de Worker (Productor) ---


# 1. Lectura de frames
def capturar_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    return cap


# 2. Preprocesamiento (resize)
def preprocesar(frame):
    # Re-escala el frame al tamaño que espera el modelo
    return cv2.resize(frame, (640, 480))


# Worker (Productor) de captura y preprocesamiento (Paralelo en CPU)
def worker_preparacion_cpu(video_path, output_queue):
    """
    Proceso de CPU dedicado a leer y preprocesar frames.
    """
    try:
        cap = capturar_video(video_path)  # Abre el video
        while True:
            ret, frame = cap.read()  # Lee un frame
            if not ret:
                output_queue.put(None)  # Envía señal de fin
                break
            processed_frame = preprocesar(frame)  # Preprocesa en CPU
            output_queue.put(processed_frame)  # Envía el frame listo a la cola
        cap.release()
    except Exception as e:
        print(f"[Productor CPU] Error: {e}")
        output_queue.put(None)


# --- Funciones de Consumidor ---


# 3. Inferencia (Paralelo en GPU)
def inferir(model, batch_de_frames):
    # Ejecuta el modelo en la GPU con un lote de frames
    return model(batch_de_frames, verbose=False, stream=False)


# 4. Postprocesamiento (en CPU, sin dibujar)
def postprocesar(frame, results):
    """
    Cuenta las detecciones. No dibuja rectángulos para no
    afectar la métrica de rendimiento puro.
    """
    rostros_detectados = 0
    for box in results.boxes:
        # Mueve los tensores de GPU a CPU
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
        conf = float(box.conf[0].cpu())
        if conf > 0.5:
            rostros_detectados += 1
    return rostros_detectados


# --- Función de Pipeline Modificada ---
def ejecutar_benchmark(video_path, model, batch_size):
    """
    Ejecuta el pipeline SIN VISUALIZACIÓN y devuelve las métricas.
    """

    # Crear cola y lanzar el proceso productor (CPU)
    preprocessed_queue = Queue(maxsize=batch_size * 2)
    prepro_process = Process(
        target=worker_preparacion_cpu,
        args=(video_path, preprocessed_queue),
    )
    prepro_process.daemon = True
    prepro_process.start()

    # Inicializar contadores
    total_frames = 0
    total_faces = 0
    tiempos_lote = []
    video_terminado = False

    # Pre-calentamiento (Warming up) de la GPU
    if batch_size > 0:
        try:
            # Enviar un lote falso para que la GPU salga del modo reposo
            dummy_frame = [np.zeros((480, 640, 3), dtype=np.uint8)] * batch_size
            _ = inferir(model, dummy_frame)
            torch.cuda.synchronize()
        except Exception:
            pass  # Ignorar errores (ej. OOM) en el calentamiento

    # Bucle principal del Consumidor
    while not video_terminado:

        # 1. Acumular un lote desde la cola
        batch_preprocesado = []
        for _ in range(batch_size):
            frame = preprocessed_queue.get()  # Espera si la cola está vacía
            if frame is None:
                video_terminado = True  # El productor envió señal de fin
                break
            batch_preprocesado.append(frame)

        if not batch_preprocesado:
            break  # Fin del video

        frames_en_lote_actual = len(batch_preprocesado)

        # --- INICIO DE MEDICIÓN ---
        start = time.time()

        # 2. Inferencia (GPU)
        results_list = inferir(model, batch_preprocesado)

        # 3. Postprocesamiento (CPU)
        rostros_en_lote = 0
        for i in range(frames_en_lote_actual):
            faces = postprocesar(None, results_list[i])
            rostros_en_lote += faces

        # 4. Sincronizar CPU/GPU
        # Fuerza al script a esperar que la GPU termine
        # para una medición de tiempo precisa.
        torch.cuda.synchronize()
        end = time.time()
        # --- FIN DE MEDICIÓN ---

        tiempos_lote.append(end - start)  # Guarda el tiempo del lote
        total_frames += frames_en_lote_actual
        total_faces += rostros_en_lote

    prepro_process.join(timeout=1)  # Limpiar el proceso productor

    if not tiempos_lote:
        return 0, 0, 0  # Evitar división por cero

    # Calcular métricas finales de esta ejecución
    duracion_total = sum(tiempos_lote)
    fps_total = total_frames / duracion_total
    latencia_frame_ms = (duracion_total / total_frames) * 1000

    return fps_total, latencia_frame_ms, total_frames


# --- Arnés de Pruebas (Test Harness) ---
if __name__ == "__main__":

    # --- Configuración de la Prueba ---
    video_path = "./videos/prueba2.mp4"  # o donde esté el video
    model_path = "../yolov8n-widerface-v2/best.pt"  # o donde esté el modelo

    # Número de veces que se repetirá la prueba completa
    N_RUNS = 3

    # Lista de BATCH SIZES a probar
    BATCHES_A_PROBAR = [1, 2, 4, 8, 16, 32, 64]

    # Pausa entre ejecuciones (segundos) para estabilización térmica
    PAUSA_TERMICA = 10

    # --- Fin de la Configuración ---

    if str(device) == "cpu":
        print("Este script de benchmark requiere CUDA. Abortando.")
        exit()

    # Cargar el modelo UNA SOLA VEZ
    print("Cargando modelo en la GPU...")
    model = YOLO(model_path)
    model.to(device)
    print("Modelo cargado.")

    # Diccionario para guardar listas de FPS
    # ej. {1: [97.1, 98.2], 2: [140.1, 142.3], ...}
    all_results = {bs: [] for bs in BATCHES_A_PROBAR}
    oom_batches = set()  # Lotes que fallaron por memoria

    # --- Bucle principal de Múltiples Ejecuciones ---
    for i in range(N_RUNS):
        print("\n" + "=" * 50)
        print(f"INICIANDO EJECUCIÓN DE PRUEBA {i + 1} DE {N_RUNS}")
        print("=" * 50)

        if i > 0:  # Pausa para estabilización térmica (excepto la primera vez)
            print(f"--- (Pausando {PAUSA_TERMICA}s para estabilización térmica...) ---")
            time.sleep(PAUSA_TERMICA)

        for bs in BATCHES_A_PROBAR:
            if bs in oom_batches:  # Omitir si ya falló por OOM
                print(f"--- Omitiendo BS = {bs} (marcado como OOM) ---")
                continue

            print(f"--- Probando BATCH_SIZE = {bs} (Run {i+1}) ---")
            try:
                # Ejecutar el benchmark
                fps, lat_ms, frames = ejecutar_benchmark(video_path, model, bs)
                if frames == 0:
                    print("Error: No se procesaron frames.")
                    all_results[bs].append(0)
                    continue

                print(f"Resultado: {fps:.2f} FPS")
                all_results[bs].append(fps)  # Guardar el resultado (FPS)

            except torch.cuda.OutOfMemoryError:
                print(f"ERROR: ¡GPU Sin Memoria (OOM) con BS={bs}!")
                oom_batches.add(bs)
                all_results[bs].append(0)  # Registrar 0 para OOM
                break  # Salir de este bucle FOR interno (no probar lotes más grandes)
            except Exception as e:
                print(f"Ocurrió un error inesperado: {e}")
                all_results[bs].append(0)
                break

    # --- FASE DE ANÁLISIS FINAL ---
    print("\n\n" + "=" * 60)
    print(f"REPORTE FINAL (PROMEDIO DE {N_RUNS} EJECUCIONES)")
    print("=" * 60)
    print(" BATCH | FPS Promedio | FPS (min) | FPS (max) | Estabilidad (StdDev)")
    print("---------------------------------------------------------")

    final_stats = {}  # Guardar estadísticas (avg, min, max, std)
    best_avg_fps = 0
    best_avg_bs = 0

    for bs in BATCHES_A_PROBAR:
        fps_list = [fps for fps in all_results[bs] if fps > 0]  # Filtrar OOMs
        if not fps_list:
            # Si todas las ejecuciones fallaron (ej. OOM), mostrar error
            print(f"  {bs:<4} |  --- OOM Error ---")
            final_stats[bs] = (0, 0, 0, float("inf"))
            continue

        # Calcular estadísticas clave
        avg_fps = np.mean(fps_list)
        min_fps = np.min(fps_list)
        max_fps = np.max(fps_list)
        std_dev = np.std(fps_list)  # Métrica clave de estabilidad

        final_stats[bs] = (avg_fps, min_fps, max_fps, std_dev)

        # Encontrar el pico de rendimiento promedio
        if avg_fps > best_avg_fps:
            best_avg_fps = avg_fps
            best_avg_bs = bs

        # Imprimir fila de la tabla
        print(
            f"  {bs:<4} | {avg_fps:>12.2f} | {min_fps:>9.2f} | {max_fps:>9.2f} | {std_dev:>10.2f}"
        )

    print("---------------------------------------------------------")
    print(
        f"🏆 Mejor promedio: {best_avg_fps:.2f} FPS con un BATCH_SIZE de {best_avg_bs}"
    )

    # --- Lógica de recomendación (Elige el más estable) ---

    # Definir "suficientemente rápido" como 95% del pico promedio
    performance_threshold = best_avg_fps * 0.95

    stable_candidates = []
    # Filtrar: solo los lotes que superan el umbral
    for bs, stats in final_stats.items():
        avg_fps, _, _, std_dev = stats
        if avg_fps >= performance_threshold:
            # (lote, estabilidad, rendimiento)
            stable_candidates.append((bs, std_dev, avg_fps))

    # Ordenar por estabilidad (std_dev más bajo primero)
    stable_candidates.sort(key=lambda x: x[1])

    # El más estable es el primero en la lista
    best_stable_bs = stable_candidates[0][0]
    best_stable_fps = stable_candidates[0][2]

    print("\n--- Recomendación ---")
    if best_stable_bs == best_avg_bs:
        print(
            f"El BATCH_SIZE {best_avg_bs} es el que tiene el mejor promedio y también es el más estable."
        )
        print(f"✅ RECOMENDACIÓN FINAL: {best_avg_bs}")
    else:
        # Esto ocurre si el pico (ej. BS=32) es inestable (alto std_dev)
        # y un lote más pequeño (ej. BS=8) es más estable.
        print(
            f"El BS {best_avg_bs} tuvo el pico ({best_avg_fps:.2f} FPS), pero es menos estable."
        )
        print(
            f"El BS {best_stable_bs} es el más estable ({best_stable_fps:.2f} FPS) dentro del 95% del rendimiento pico."
        )
        print(f"✅ RECOMENDACIÓN FINAL (más segura): {best_stable_bs}")
