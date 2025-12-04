import customtkinter as ctk
import cv2
# import threading
import time
import random
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

# --- Configuración de la Interfaz ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


class VigilanceApp(ctk.CTk):
    """
    Aplicación de Escritorio para Procesamiento de Video con Detección Facial.
    Permite cargar un video, procesarlo frame a frame y medir el tiempo total.
    """

    def __init__(self):
        super().__init__()

        # --- Variables de Estado ---
        self.processing = False
        self.cap = None
        self.video_path = None
        self.start_time = 0
        self.frame_counter = 0
        self.total_frames = 0

        # Intentamos cargar el modelo real al iniciar
        self.model = self.load_your_model()

        # --- Configuración de la Ventana Principal ---
        self.title("Sistema de Vigilancia: Procesamiento de Video")
        self.geometry("900x750")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- 1. Título (Fila 0) ---
        self.title_frame = ctk.CTkFrame(self, corner_radius=10)
        self.title_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        ctk.CTkLabel(self.title_frame, text="Procesador de Video con Detección Facial (YOLO)",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(padx=10, pady=10)

        # --- 2. Área de Video (Fila 1) ---
        self.video_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1a1a1a")  # Color oscuro
        self.video_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_frame, text="Selecciona un video para comenzar", text_color="gray",
                                        font=ctk.CTkFont(size=20, weight="bold"))
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # --- 3. Controles y Métricas (Fila 2) ---
        self.control_frame = ctk.CTkFrame(self, corner_radius=10)
        self.control_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.control_frame.columnconfigure((0, 1, 2, 3), weight=1)

        # Botón Seleccionar Video
        self.btn_select = ctk.CTkButton(self.control_frame, text="📂 Cargar Video", command=self.select_video,
                                        font=ctk.CTkFont(size=14), fg_color="#2b2b2b", hover_color="#404040")
        self.btn_select.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Botón Iniciar Procesamiento
        self.btn_start = ctk.CTkButton(self.control_frame, text="▶ Iniciar Procesamiento",
                                       command=self.start_processing,
                                       font=ctk.CTkFont(size=14, weight="bold"), state="disabled")
        self.btn_start.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Etiquetas de Estado
        self.lbl_status = ctk.CTkLabel(self.control_frame, text="Estado: En espera", text_color="gray")
        self.lbl_status.grid(row=0, column=2, padx=10, sticky="w")

        self.lbl_counter = ctk.CTkLabel(self.control_frame, text="Frames: 0/0")
        self.lbl_counter.grid(row=0, column=3, padx=10, sticky="e")

        # --- 4. Log (Fila 3) ---
        self.log_text = ctk.CTkTextbox(self, height=80, corner_radius=10)
        self.log_text.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.log_event("Sistema iniciado. Por favor cargue un archivo de video.", "info")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ---------------------------------------------------------
    #  ZONA DE INTEGRACIÓN DE TU MODELO (YOLO + PARALELISMO)
    # ---------------------------------------------------------
    @staticmethod
    def load_your_model():
        """
        Aquí es donde debes importar e inicializar tu modelo.
        """
        try:
            # EJEMPLO DE IMPORTACIÓN (Descomenta y ajusta según tu código):
            # from ultralytics import YOLO
            # model = YOLO("yolov8n.pt")
            # print("Modelo cargado exitosamente")
            # return model
            return None  # Retorna None porque es una simulación
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return None

    def detect_faces(self, frame, frame_number):
        """
        Recibe un frame (imagen) y retorna las detecciones.
        Sustituye la lógica simulada por tu llamada a YOLO.
        """
        height, width, _ = frame.shape
        detections = []

        # --- OPCIÓN A: USAR TU MODELO REAL ---
        if self.model is not None:
            # results = self.model(frame)
            # Procesar 'results' para obtener formato: [{'x': 10, 'y': 10, 'w': 50, 'h': 50}, ...]
            # detections = ...
            pass

        # --- OPCIÓN B: SIMULACIÓN (Para el demo) ---
        else:
            # Simulamos carga de procesamiento (pequeño delay para simular inferencia)
            # time.sleep(0.01)
            if frame_number % 5 == 0:  # Detectar cada 5 frames para variar
                detections = self.simulate_dummy_detection(width, height)

        return detections

    @staticmethod
    def simulate_dummy_detection(width, height):
        # Genera cajas aleatorias para simular que el modelo encontró algo
        num = random.randint(0, 2)
        dets = []
        for _ in range(num):
            dets.append({
                'x': random.randint(0, width - 100), 'y': random.randint(0, height - 100),
                'w': random.randint(50, 150), 'h': random.randint(50, 150),
                'conf': round(random.uniform(0.8, 0.99), 2)
            })
        return dets

    # ---------------------------------------------------------

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de Video", "*.mp4 *.avi *.mkv *.mov")])
        if file_path:
            self.video_path = file_path
            self.log_event(f"Video cargado: {file_path.split('/')[-1]}", "info")
            self.btn_start.configure(state="normal", fg_color="#0d6efd")  # Azul habilitado
            self.lbl_status.configure(text="Estado: Listo para iniciar", text_color="#0d6efd")

            # Obtener info básica del video
            temp_cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            temp_cap.release()
            self.lbl_counter.configure(text=f"Frames: 0/{self.total_frames}")

    def start_processing(self):
        if not self.video_path:
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.log_event("Error al abrir el archivo de video.", "error")
            return

        self.processing = True
        self.frame_counter = 0
        self.start_time = time.time()  # Iniciamos el cronómetro

        # Actualizar UI
        self.btn_select.configure(state="disabled")
        self.btn_start.configure(text="⏹ Detener", command=self.stop_processing, fg_color="#dc3545",
                                 hover_color="#a71d2a")
        self.lbl_status.configure(text="Estado: Procesando...", text_color="#28a745")
        self.log_event("Procesamiento iniciado...", "info")

        # Iniciar bucle
        self.process_loop()

    def stop_processing(self):
        self.processing = False
        if self.cap:
            self.cap.release()

        # Reset UI
        self.btn_select.configure(state="normal")
        self.btn_start.configure(text="▶ Iniciar Procesamiento", command=self.start_processing, fg_color="#0d6efd",
                                 hover_color="#0056b3")
        self.lbl_status.configure(text="Estado: Detenido", text_color="gray")
        self.video_label.configure(image=None, text="Procesamiento finalizado o detenido.")

    def finish_processing_success(self):
        """Se llama cuando el video termina naturalmente."""
        end_time = time.time()
        total_time = end_time - self.start_time

        self.stop_processing()
        self.log_event(f"Finalizado. Tiempo total: {total_time:.2f} segundos.", "info")

        # Mostrar alerta con el tiempo
        messagebox.showinfo("Procesamiento Completado",
                            f"Video procesado exitosamente.\n\n"
                            f"Tiempo Total: {total_time:.4f} segundos\n"
                            f"Total Frames: {self.frame_counter}")

    def process_loop(self):
        if not self.processing:
            return

        ret, frame = self.cap.read()
        if not ret:
            # Fin del video
            self.finish_processing_success()
            return

        self.frame_counter += 1

        # --- PASO 1: DETECCIÓN (Donde actúa tu modelo) ---
        detections = self.detect_faces(frame, self.frame_counter)

        # --- PASO 2: DIBUJAR RESULTADOS ---
        # Dibujamos directamente sobre el frame
        for d in detections:
            x, y, w, h = d['x'], d['y'], d['w'], d['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {d['conf']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- PASO 3: MOSTRAR EN GUI ---
        # Convertir color y formato para Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Ajustar tamaño al contenedor (manteniendo proporción)
        container_w = self.video_frame.winfo_width()
        container_h = self.video_frame.winfo_height()
        if container_w > 10 and container_h > 10:
            pil_img.thumbnail((container_w, container_h), Image.Resampling.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil_img)
        self.video_label.configure(image=tk_img, text="")
        self.video_label.image = tk_img  # Referencia para evitar Garbage Collection

        # Actualizar contadores
        if self.frame_counter % 5 == 0:  # Actualizar texto cada 5 frames para no saturar
            self.lbl_counter.configure(text=f"Frames: {self.frame_counter}/{self.total_frames}")

        # Llamar al siguiente frame lo más rápido posible (1ms)
        self.after(1, self.process_loop)

    def log_event(self, message, type="info"):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("0.0", f"[{timestamp}] {message}\n")

    def on_closing(self):
        self.stop_processing()
        self.destroy()


if __name__ == "__main__":
    app = VigilanceApp()
    app.mainloop()


# ¿Cómo importar tu modelo?
"""
En el código he dejado una función llamada `load_your_model` y otra llamada `detect_faces`.Aquí es
donde debes hacer la conexión. Si tienes tu modelo en un archivo separado(digamos `mi_modelo_paralelo.py`
y tienes una clase o función que detecta, harías lo siguiente:

1. ** Importar: ** Al inicio del archivo `security_app.py`, añade:
```python
from mi_modelo_paralelo import DetectorFacial  # Tu clase

```

2. ** Inicializar: ** Modifica el método `load_your_model`:
```python
def load_your_model(self):
    print("Cargando modelo YOLO paralelo...")
    return DetectorFacial(weights="best.pt")  # Inicializa tu clase
```

3. ** Detectar: ** Modifica el método `detect_faces` en la sección "OPCIÓN A":
```python

def detect_faces(self, frame, frame_number):
    # ...
    if self.model is not None:
        # Asumiendo que tu modelo tiene un método .predict(imagen)
        # que devuelve una lista de diccionarios [{'x':.., 'y':..}, ...]
        detections = self.model.predict(frame)
        return detections
    # ...
    """