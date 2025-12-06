# Procesamiento de imagenes
Desarrollo de un pipeline híbrido paralelo basado en el patrón Productor-Consumidor, que utiliza TensorRT y Batching (procesamiento por lotes de 64 frames) para superar los cuellos de botella de comunicación entre CPU y GPU (PCIe y Kernel Launch Overhead). Gracias a la optimización con precisión FP16 y el uso de CuPy para el postprocesamiento en memoria de la GPU (Zero-Copy).

### Integrantes
- Chávez Ccahuana, Álvaro Andrés
- Jara Espinoza, Angela
- Obando Salinas, Enmanuel José
- Patricio Julca, Vilberto Alberto
- Pumapillo Sarmiento, Bruno
- Vera Alva, Miguel Angel

### Librerías utilizadas
- **YOLO v8 nano:** El modelo de detección de objetos base, elegido por ser la versión más ligera, lo que permite maximizar el tamaño del lote (batch size) para una mayor eficiencia de la GPU.
- **TensorRT (NVIDIA):** Se usa para optimizar el modelo entrenado y compilarlo a un motor binario que aprovecha la arquitectura de la GPU, incluyendo la reducción de precisión a FP16 para activar los Tensor Cores.
- **OpenCV:** Empleada para la ingesta del video, es decir, para leer los frames del archivo o cámara y convertirlos en matrices numéricas.
- **Ultralytics YOLO:** La librería que proporciona la lógica central del algoritmo de detección.
- **Python Threading:** Se utiliza para implementar el patrón Productor-Consumidor y realizar la lectura y la inferencia en hilos distintos, evitando bloqueos.
- **Python CuPy:** Utilizado para la innovación Zero-Copy, permitiendo manipular los resultados de la detección (matrices) directamente en la memoria de la GPU (VRAM) para evitar el cuello de botella de la transferencia de datos a la CPU.
