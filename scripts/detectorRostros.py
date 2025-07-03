from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import os
import numpy as np

def verificar_modelo():
    """Verificar y cargar el modelo desde Hugging Face o local"""
    
    modelo_local = '../yolov8n-widerface-v2/best.pt'
    
    if os.path.exists(modelo_local):
        print(f"Cargando modelo local desde {modelo_local}...")
        model = YOLO(modelo_local)
        modelo_path = modelo_local
    else:
        print(f"Descargando modelo desde Hugging Face...")
        
        try:
            modelo_path = hf_hub_download(
                repo_id="arnabdhar/YOLOv8-Face-Detection", 
                filename="model.pt"
            )
            
            if not os.path.exists(modelo_path):
                print(f"Error: No se pudo descargar el modelo")
                return None
            
            model = YOLO(modelo_path)
            
        except Exception as e:
            print(f"Error al descargar el modelo: {str(e)}")
            return None
    
    print(f"Modelo cargado exitosamente")
    print(f"Clases: {model.names}")
    
    return model

def procesar_video():
    """Procesar video con detección de rostros"""
    model = verificar_modelo()
    if model is None:
        return
    
    video_path = None
    posibles_videos = ['videos/prueba2.mp4']
    
    for path in posibles_videos:
        if os.path.exists(path):
            video_path = path
            break
    
    if video_path is None:
        video_path = input("Ingresa la ruta del video: ").strip()
        if not os.path.exists(video_path):
            print("Video no encontrado")
            return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se puede abrir el video")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {fps} FPS, {total_frames} frames, {width}x{height}")
    
    frame_count = 0
    rostros_total = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        results = model(frame, 
                       verbose=False, 
                       conf=0.5,
                       iou=0.3,
                       agnostic_nms=True)
        
        rostros_en_frame = 0
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                if confidence >= 0.5:
                    color = (0, 255, 0)
                    
                    if any(keyword in class_name.lower() for keyword in ['face', 'rostro', 'cara']):
                        rostros_en_frame += 1
                    elif 'person' in class_name.lower():
                        color = (255, 0, 0)
                    else:
                        color = (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        rostros_total += rostros_en_frame
        
        info_lines = [
            f'Frame: {frame_count}/{total_frames}',
            f'Rostros: {rostros_en_frame} | Total: {rostros_total}'
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + (i * 25)
            cv2.putText(frame, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Detección de Rostros - Video', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord(' '):
            cv2.waitKey(0)
        elif key == ord('s'):
            filename = f'video_frame_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Frame guardado como {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Frames procesados: {frame_count}")
    print(f"Rostros detectados: {rostros_total}")

def procesar_imagen():
    """Procesar imagen con detección de rostros"""
    model = verificar_modelo()
    if model is None:
        return
    
    imagen_path = None
    posibles_imagenes = []
    
    if os.path.exists('images'):
        for archivo in os.listdir('images'):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                posibles_imagenes.append(os.path.join('images', archivo))
    
    if posibles_imagenes:
        print(f"Imágenes encontradas:")
        for i, img in enumerate(posibles_imagenes, 1):
            print(f"   {i}. {img}")
        
        try:
            seleccion = int(input(f"Selecciona imagen (1-{len(posibles_imagenes)}): "))
            if 1 <= seleccion <= len(posibles_imagenes):
                imagen_path = posibles_imagenes[seleccion-1]
            else:
                print("Selección inválida")
                return
        except ValueError:
            print("Número inválido")
            return
    else:
        imagen_path = input("Ingresa la ruta de la imagen: ").strip()
        if not os.path.exists(imagen_path):
            print("Imagen no encontrada")
            return
    
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        print("Error: No se puede cargar la imagen")
        return
    
    altura, ancho = imagen.shape[:2]
    print(f"Dimensiones: {ancho}x{altura} píxeles")
    
    results = model(imagen, 
                   verbose=False, 
                   conf=0.5,
                   iou=0.3,
                   agnostic_nms=True)
    
    num_detecciones = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"Detecciones encontradas: {num_detecciones}")
    
    if num_detecciones > 0:
        imagen_resultado = imagen.copy()
        rostros_detectados = 0
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            if any(keyword in class_name.lower() for keyword in ['face', 'rostro', 'cara', 'head']):
                color = (0, 255, 0)
                rostros_detectados += 1
            elif 'person' in class_name.lower():
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            
            cv2.rectangle(imagen_resultado, (x1, y1), (x2, y2), color, 3)
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(imagen_resultado, label, (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            print(f"{class_name}: {confidence:.3f} en [{x1},{y1},{x2},{y2}]")
        
        nombre_archivo = os.path.basename(imagen_path).split('.')[0]
        output_filename = f'resultado_{nombre_archivo}.jpg'
        cv2.imwrite(output_filename, imagen_resultado)
        
        print(f"Resultado guardado como: {output_filename}")
        print(f"Rostros detectados: {rostros_detectados}")
        
        cv2.imshow('Detección de Rostros - Imagen', imagen_resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No se detectó nada en la imagen")

def procesar_camara():
    """Procesar video en tiempo real desde la cámara"""
    model = verificar_modelo()
    if model is None:
        return
    
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    total_rostros = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer de la cámara")
            break
        
        frame_count += 1
        
        if frame_count % 3 == 0:
            results = model(frame, 
                           verbose=False, 
                           conf=0.6,
                           iou=0.25,
                           max_det=8,
                           agnostic_nms=True)
            
            rostros_en_frame = 0
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    if any(keyword in class_name.lower() for keyword in ['face', 'rostro', 'cara', 'head']):
                        color = (0, 255, 0)
                        rostros_en_frame += 1
                    elif 'person' in class_name.lower():
                        color = (255, 0, 0)
                    else:
                        color = (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            total_rostros += rostros_en_frame
            
            info_lines = [
                f'Frame: {frame_count}',
                f'Rostros: {rostros_en_frame} | Total: {total_rostros}'
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 30 + (i * 25)
                cv2.putText(frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Detección de Rostros - Cámara', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord(' '):
            cv2.waitKey(0)
        elif key == ord('s'):
            filename = f'camara_foto_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Foto guardada como {filename}")
        elif key == ord('c'):
            cap.release()
            camera_id = 1 if camera_id == 0 else 0
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                camera_id = 0
                cap = cv2.VideoCapture(camera_id)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Frames procesados: {frame_count}")
    print(f"Total de rostros detectados: {total_rostros}")

def procesar_reconocimiento():
    """Reconocimiento facial usando LBPH + detección con YOLO"""
    model_yolo = verificar_modelo()
    if model_yolo is None:
        return

    dataPath = '../lbph_data/faces'
    modelPath = '../lbph_data/lbph_model.xml'
    if not os.path.exists(modelPath):
        print(f"Modelo LBPH no encontrado en {modelPath}")
        return

    imagePaths = os.listdir(dataPath)
    print(f"Etiquetas: {imagePaths}")

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(modelPath)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolo(frame, 
                            verbose=False, 
                            conf=0.7,
                            iou=0.25,
                            max_det=5,
                            agnostic_nms=True)

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, frame.shape[1]), min(y2, frame.shape[0])
                
                rostro = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray, (150, 150), interpolation=cv2.INTER_CUBIC)

                label, confidence = face_recognizer.predict(gray_resized)

                if confidence < 74:
                    nombre = imagePaths[label]
                    color = (0, 255, 0)
                else:
                    nombre = "Desconocido"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{nombre} ({confidence:.1f})', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Reconocimiento Facial LBPH', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def menu_principal():
    """Menú principal de la aplicación"""
    print("DETECTOR DE ROSTROS CON YOLO")
    print("=" * 50)
    
    while True:
        print(f"\nMENU PRINCIPAL:")
        print(f"1. Procesar Video")
        print(f"2. Procesar Imagen")
        print(f"3. Usar Cámara en Tiempo Real")
        print(f"4. Información del Modelo")
        print(f"5. Reconocimiento Facial LBPH")
        print(f"6. Salir")
        
        opcion = input(f"\nSelecciona una opción (1-6): ").strip()
        
        if opcion == "1":
            procesar_video()
        elif opcion == "2":
            procesar_imagen()
        elif opcion == "3":
            procesar_camara()
        elif opcion == "4":
            verificar_modelo()
        elif opcion == "5":
            procesar_reconocimiento()
        elif opcion == "6":
            print(f"Hasta luego!")
            break
        else:
            print(f"Opción inválida. Selecciona 1-6.")
        
        input(f"\nPresiona Enter para continuar...")

if __name__ == "__main__":
    menu_principal()