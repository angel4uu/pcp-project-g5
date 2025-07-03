from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import os
import numpy as np

def verificar_modelo():
    """Verificar y cargar el modelo desde Hugging Face o local"""
    
    # Primero intentar cargar modelo local  
    modelo_local = '../yolov8n-widerface-v1/best.pt'
    
    if os.path.exists(modelo_local):
        print(f"🔄 Cargando modelo local desde {modelo_local}...")
        model = YOLO(modelo_local)
        modelo_path = modelo_local
    else:
        print(f"❌ No se encuentra modelo local en {modelo_local}")
        print(f"🔄 Descargando modelo de detección de rostros desde Hugging Face...")
        
        try:
            # Descargar modelo desde Hugging Face
            modelo_path = hf_hub_download(
                repo_id="arnabdhar/YOLOv8-Face-Detection", 
                filename="model.pt"
            )
            
            if not os.path.exists(modelo_path):
                print(f"❌ Error: No se pudo descargar el modelo")
                return None
            
            print(f"✅ Modelo descargado en: {modelo_path}")
            print(f"🔄 Cargando modelo...")
            model = YOLO(modelo_path)
            
        except Exception as e:
            print(f"❌ Error al descargar el modelo: {str(e)}")
            return None
    
    # Verificar información del modelo
    print(f"✅ Modelo cargado exitosamente")
    print(f"📋 Información del modelo:")
    print(f"   - Clases entrenadas: {model.names}")
    print(f"   - Número de clases: {len(model.names)}")
    
    # Verificar clases relacionadas con rostros
    face_related_classes = []
    for idx, name in model.names.items():
        if any(keyword in name.lower() for keyword in ['face', 'rostro', 'cara', 'person', 'head']):
            face_related_classes.append((idx, name))
    
    if face_related_classes:
        print(f"✅ Clases relacionadas con rostros: {face_related_classes}")
    else:
        print("⚠️  ADVERTENCIA: No se encontraron clases relacionadas con rostros")
        print("   El modelo puede no estar optimizado para detección de rostros")
    
    return model

def procesar_video():
    """Procesar video con detección de rostros"""
    model = verificar_modelo()
    if model is None:
        return
    
    # Buscar video
    video_path = None
    posibles_videos = ['videos/prueba2.mp4']
    
    print(f"\n🔍 Buscando video...")
    for path in posibles_videos:
        if os.path.exists(path):
            video_path = path
            break
    
    if video_path is None:
        print("❌ No se encontró ningún video en la carpeta videos/")
        video_path = input("Ingresa la ruta completa del video: ").strip()
        if not os.path.exists(video_path):
            print("❌ Video no encontrado")
            return
    
    print(f"✅ Video encontrado: {video_path}")
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: No se puede abrir el video")
        return
    
    # Info del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video: {fps} FPS, {total_frames} frames, {width}x{height}")
    print(f"Controles: ESC=Salir, ESPACIO=Pausar, S=Guardar frame")
    
    frame_count = 0
    rostros_total = 0
    detecciones_totales = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detectar
        results = model(frame, verbose=False, conf=0.1)
        todas_detecciones = len(results[0].boxes) if results[0].boxes is not None else 0
        detecciones_totales += todas_detecciones
        
        # Procesar detecciones
        rostros_en_frame = 0
        detecciones_mostradas = 0
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                if confidence > 0.1:
                    color = (0, 255, 0)  # Verde por defecto
                    
                    # Cambiar color según la clase
                    if any(keyword in class_name.lower() for keyword in ['face', 'rostro', 'cara']):
                        color = (0, 255, 0)  # Verde para rostros
                        rostros_en_frame += 1
                    elif 'person' in class_name.lower():
                        color = (255, 0, 0)  # Azul para personas
                    else:
                        color = (0, 0, 255)  # Rojo para otras clases
                    
                    # Dibujar
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    detecciones_mostradas += 1
        
        rostros_total += rostros_en_frame
        
        # Info en pantalla
        info_lines = [
            f'Frame: {frame_count}/{total_frames}',
            f'Detecciones: {todas_detecciones} | Mostradas: {detecciones_mostradas}',
            f'Rostros: {rostros_en_frame} | Total: {rostros_total}'
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + (i * 25)
            cv2.putText(frame, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar frame
        cv2.imshow('Detección de Rostros - Video', frame)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # ESPACIO
            print("⏸️  Pausado - Presiona cualquier tecla para continuar")
            cv2.waitKey(0)
        elif key == ord('s'):  # S
            filename = f'video_frame_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"💾 Frame guardado como {filename}")
    
    # Estadísticas finales
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ ¡Análisis de video completado!")
    print(f"📊 Estadísticas:")
    print(f"   - Frames procesados: {frame_count}")
    print(f"   - Detecciones totales: {detecciones_totales}")
    print(f"   - Rostros detectados: {rostros_total}")
    if frame_count > 0:
        print(f"   - Promedio detecciones/frame: {detecciones_totales/frame_count:.2f}")
        print(f"   - Promedio rostros/frame: {rostros_total/frame_count:.2f}")

def procesar_imagen():
    """Procesar imagen con detección de rostros"""
    model = verificar_modelo()
    if model is None:
        return
    
    # Buscar imagen
    imagen_path = None
    posibles_imagenes = []
    
    # Buscar en carpeta images
    if os.path.exists('images'):
        for archivo in os.listdir('images'):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                posibles_imagenes.append(os.path.join('images', archivo))
    
    if posibles_imagenes:
        print(f"\n📁 Imágenes encontradas:")
        for i, img in enumerate(posibles_imagenes, 1):
            print(f"   {i}. {img}")
        
        try:
            seleccion = int(input(f"Selecciona imagen (1-{len(posibles_imagenes)}): "))
            if 1 <= seleccion <= len(posibles_imagenes):
                imagen_path = posibles_imagenes[seleccion-1]
            else:
                print("❌ Selección inválida")
                return
        except ValueError:
            print("❌ Por favor ingresa un número válido")
            return
    else:
        print("❌ No se encontraron imágenes en la carpeta images/")
        imagen_path = input("Ingresa la ruta completa de la imagen: ").strip()
        if not os.path.exists(imagen_path):
            print("❌ Imagen no encontrada")
            return
    
    print(f"✅ Imagen seleccionada: {imagen_path}")
    
    # Cargar imagen
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        print("❌ Error: No se puede cargar la imagen")
        return
    
    altura, ancho = imagen.shape[:2]
    print(f"📊 Dimensiones: {ancho}x{altura} píxeles")
    
    # Probar con diferentes umbrales
    umbrales = [0.05, 0.1, 0.25, 0.5]
    mejor_resultado = None
    max_detecciones = 0
    
    print(f"\n🎯 Probando con diferentes umbrales...")
    
    for umbral in umbrales:
        results = model(imagen, verbose=False, conf=umbral)
        num_detecciones = len(results[0].boxes) if results[0].boxes is not None else 0
        
        if num_detecciones > max_detecciones:
            max_detecciones = num_detecciones
            mejor_resultado = (umbral, results)
        
        print(f"   Umbral {umbral}: {num_detecciones} detecciones")
    
    if mejor_resultado is not None:
        umbral_optimo, results_optimo = mejor_resultado
        imagen_resultado = imagen.copy()
        
        rostros_detectados = 0
        
        if results_optimo[0].boxes is not None:
            for i, box in enumerate(results_optimo[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Color según tipo
                if any(keyword in class_name.lower() for keyword in ['face', 'rostro', 'cara', 'head']):
                    color = (0, 255, 0)  # Verde para rostros
                    rostros_detectados += 1
                elif 'person' in class_name.lower():
                    color = (255, 0, 0)  # Azul para personas
                else:
                    color = (0, 0, 255)  # Rojo para otros
                
                # Dibujar
                cv2.rectangle(imagen_resultado, (x1, y1), (x2, y2), color, 3)
                label = f'{class_name} {confidence:.2f}'
                cv2.putText(imagen_resultado, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                print(f"   🔍 {class_name}: {confidence:.3f} en [{x1},{y1},{x2},{y2}]")
        
        # Guardar resultado
        nombre_archivo = os.path.basename(imagen_path).split('.')[0]
        output_filename = f'resultado_{nombre_archivo}.jpg'
        cv2.imwrite(output_filename, imagen_resultado)
        
        print(f"\n✅ Resultado guardado como: {output_filename}")
        print(f"👤 Rostros detectados: {rostros_detectados}")
        
        # Mostrar imagen
        print(f"👁️  Mostrando resultado - Presiona cualquier tecla para cerrar")
        cv2.imshow('Detección de Rostros - Imagen', imagen_resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ No se detectó nada en la imagen")

def procesar_camara():
    """Procesar video en tiempo real desde la cámara"""
    model = verificar_modelo()
    if model is None:
        return
    
    print(f"\n📹 Iniciando cámara...")
    print(f"Controles: ESC=Salir, ESPACIO=Pausar, S=Guardar foto, C=Cambiar cámara")
    
    # Intentar abrir cámara
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Error: No se puede abrir la cámara")
        return
    
    # Configurar cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    total_rostros = 0
    
    print(f"✅ Cámara iniciada correctamente")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al leer de la cámara")
            break
        
        frame_count += 1
        
        # Detectar cada 3 frames para mejor rendimiento
        if frame_count % 3 == 0:
            results = model(frame, verbose=False, conf=0.3)
            
            rostros_en_frame = 0
            detecciones_totales = 0
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                detecciones_totales = len(results[0].boxes)
                
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Color según tipo
                    if any(keyword in class_name.lower() for keyword in ['face', 'rostro', 'cara', 'head']):
                        color = (0, 255, 0)  # Verde para rostros
                        rostros_en_frame += 1
                    elif 'person' in class_name.lower():
                        color = (255, 0, 0)  # Azul para personas
                    else:
                        color = (0, 0, 255)  # Rojo para otros
                    
                    # Dibujar
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            total_rostros += rostros_en_frame
            
            # Info en pantalla
            info_lines = [
                f'Frame: {frame_count}',
                f'Detecciones: {detecciones_totales}',
                f'Rostros: {rostros_en_frame} | Total: {total_rostros}',
                f'Camara: {camera_id}'
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 30 + (i * 25)
                cv2.putText(frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar frame
        cv2.imshow('Detección de Rostros - Cámara', frame)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # ESPACIO
            print("⏸️  Pausado - Presiona cualquier tecla para continuar")
            cv2.waitKey(0)
        elif key == ord('s'):  # S
            filename = f'camara_foto_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"📸 Foto guardada como {filename}")
        elif key == ord('c'):  # C
            cap.release()
            camera_id = 1 if camera_id == 0 else 0
            print(f"🔄 Cambiando a cámara {camera_id}")
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                print(f"❌ No se puede abrir cámara {camera_id}, regresando a cámara 0")
                camera_id = 0
                cap = cv2.VideoCapture(camera_id)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Sesión de cámara terminada")
    print(f"📊 Frames procesados: {frame_count}")
    print(f"👤 Total de rostros detectados: {total_rostros}")

def menu_principal():
    """Menú principal de la aplicación"""
    print("🤖 DETECTOR DE ROSTROS CON YOLO")
    print("=" * 50)
    
    while True:
        print(f"\n📋 MENÚ PRINCIPAL:")
        print(f"1. 🎥 Procesar Video")
        print(f"2. 🖼️  Procesar Imagen")
        print(f"3. 📹 Usar Cámara en Tiempo Real")
        print(f"4. ℹ️  Información del Modelo")
        print(f"5. 🚪 Salir")
        
        opcion = input(f"\nSelecciona una opción (1-5): ").strip()
        
        if opcion == "1":
            print(f"\n" + "="*30 + " MODO VIDEO " + "="*30)
            procesar_video()
        
        elif opcion == "2":
            print(f"\n" + "="*30 + " MODO IMAGEN " + "="*30)
            procesar_imagen()
        
        elif opcion == "3":
            print(f"\n" + "="*30 + " MODO CÁMARA " + "="*30)
            procesar_camara()
        
        elif opcion == "4":
            print(f"\n" + "="*25 + " INFORMACIÓN DEL MODELO " + "="*25)
            verificar_modelo()
        
        elif opcion == "5":
            print(f"\n👋 ¡Hasta luego!")
            break
        
        else:
            print(f"❌ Opción inválida. Por favor selecciona 1-5.")
        
        input(f"\nPresiona Enter para continuar...")

if __name__ == "__main__":
    menu_principal()