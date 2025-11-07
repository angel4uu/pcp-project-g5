#!/usr/bin/env python3
"""
Ejemplo completo de cómo usar HU-04: Optimización de Inferencia

Este script demuestra:
1. Cargar modelo PyTorch (baseline)
2. Exportar a ONNX
3. Comparar velocidad (benchmarking)
4. Validar precisión
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

def example_1_load_pytorch_model():
    """Ejemplo 1: Cargar modelo PyTorch."""
    print("\n" + "=" * 70)
    print("EJEMPLO 1: Cargar modelo PyTorch")
    print("=" * 70)
    
    print("\n📖 Código:")
    print("""
from ultralytics import YOLO
import torch

# Verificar CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando: {device}")

# Cargar modelo
model = YOLO('model.pt')
model.to(device)

# Hacer inferencia
results = model('image.jpg', verbose=False)
print(f"Detecciones: {len(results[0].boxes)}")
    """)
    
    print("\n⏱️  Tiempo esperado: 45 ms/frame (CPU), 15 ms/frame (GPU)")
    print("📊 Output: resultados con bounding boxes y confianzas")

def example_2_export_to_onnx():
    """Ejemplo 2: Exportar a ONNX."""
    print("\n" + "=" * 70)
    print("EJEMPLO 2: Exportar modelo YOLO a ONNX")
    print("=" * 70)
    
    print("\n📖 Código:")
    print("""
from ultralytics import YOLO
import torch

# Cargar modelo
model = YOLO('model.pt')

# Exportar a ONNX
exported_path = model.export(
    format='onnx',
    opset=12,
    simplify=True,
    device=0 if torch.cuda.is_available() else 'cpu'
)
print(f"✅ Exportado a: {exported_path}")
    """)
    
    print("\n⏱️  Tiempo esperado: 2-5 minutos (solo una vez)")
    print("📊 Output: archivo models/model.onnx (~48 MB)")
    print("💡 Ventaja: formato interoperable, 1.5x rápido vs PyTorch")

def example_3_benchmark():
    """Ejemplo 3: Benchmarking PyTorch vs ONNX."""
    print("\n" + "=" * 70)
    print("EJEMPLO 3: Benchmarking - PyTorch vs ONNX")
    print("=" * 70)
    
    print("\n📖 Código:")
    print("""
import numpy as np
import time
from ultralytics import YOLO
import onnxruntime as ort

# Setup
model = YOLO('model.pt')
onnx_session = ort.InferenceSession(
    'models/model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Benchmark PyTorch
pytorch_times = []
for _ in range(100):
    start = time.perf_counter()
    _ = model(dummy_input)
    pytorch_times.append((time.perf_counter() - start) * 1000)

# Benchmark ONNX
onnx_times = []
for _ in range(100):
    start = time.perf_counter()
    input_name = onnx_session.get_inputs()[0].name
    _ = onnx_session.run(None, {input_name: dummy_input})
    onnx_times.append((time.perf_counter() - start) * 1000)

# Resultados
pytorch_avg = np.mean(pytorch_times)
onnx_avg = np.mean(onnx_times)
speedup = pytorch_avg / onnx_avg

print(f"PyTorch: {pytorch_avg:.2f} ms")
print(f"ONNX:    {onnx_avg:.2f} ms")
print(f"Speedup: {speedup:.2f}x")
    """)
    
    print("\n📊 Output esperado:")
    print("   PyTorch: 45.32 ms")
    print("   ONNX:    18.45 ms")
    print("   Speedup: 2.46x ✅")

def example_4_validate_precision():
    """Ejemplo 4: Validar precisión (similitud outputs)."""
    print("\n" + "=" * 70)
    print("EJEMPLO 4: Validar Precisión (mAP)")
    print("=" * 70)
    
    print("\n📖 Concepto:")
    print("""
La precisión se mide comparando:
- PyTorch (baseline): outputs "verdaderos"
- ONNX (exportado): outputs a validar

Métrica: mAP (mean Average Precision)
- rango: 0-100%
- aceptable: >98% similitud vs PyTorch
    """)
    
    print("\n📖 Código:")
    print("""
from ultralytics import YOLO
import onnxruntime as ort
import cv2
import numpy as np

model = YOLO('model.pt')
onnx_session = ort.InferenceSession('models/model.onnx')

# Comparar en 10 imágenes de prueba
similarities = []
for image_path in image_paths[:10]:
    # PyTorch
    pytorch_results = model(image_path, verbose=False)
    pytorch_dets = len(pytorch_results[0].boxes)
    
    # ONNX
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))[np.newaxis, :]
    
    input_name = onnx_session.get_inputs()[0].name
    onnx_outputs = onnx_session.run(None, {input_name: image})
    onnx_dets = len(onnx_outputs[0]) if onnx_outputs else 0
    
    similarity = 1.0 - abs(pytorch_dets - onnx_dets) / max(pytorch_dets, onnx_dets, 1)
    similarities.append(similarity)

avg_similarity = np.mean(similarities) * 100
print(f"Similitud promedio: {avg_similarity:.2f}%")
print(f"Status: {'✅ ACEPTADO' if avg_similarity > 98 else '❌ REVISAR'}")
    """)
    
    print("\n📊 Output esperado:")
    print("   Similitud: 98.50%")
    print("   Status: ✅ ACEPTADO")

def example_5_tensorrt_export():
    """Ejemplo 5: Exportar ONNX a TensorRT (avanzado)."""
    print("\n" + "=" * 70)
    print("EJEMPLO 5: Exportar ONNX a TensorRT (Avanzado)")
    print("=" * 70)
    
    print("\n📖 Requisitos:")
    print("""
- TensorRT 8.6+ instalado
- CUDA 11.8+ disponible
- trtexec en PATH
    """)
    
    print("\n📖 Comando:")
    print("""
trtexec --onnx=models/model.onnx \\
        --saveEngine=models/model.fp16.engine \\
        --workspace=1024 \\
        --fp16
    """)
    
    print("\n⏱️  Tiempo esperado: 5-15 minutos")
    print("📊 Output: archivo models/model.fp16.engine (~15 MB)")
    print("💡 Ventaja: 3-5x rápido vs PyTorch, solo GPU Nvidia")

def example_6_cpp_tensorrt():
    """Ejemplo 6: Usar TensorRT desde C++ (avanzado)."""
    print("\n" + "=" * 70)
    print("EJEMPLO 6: Pipeline C++ + TensorRT + CUDA (Avanzado)")
    print("=" * 70)
    
    print("\n📖 Compilación:")
    print("""
cd scripts
mkdir build && cd build
cmake .. -DTENSORRT_ROOT=/path/to/tensorrt
cmake --build . --config Release -j8
    """)
    
    print("\n📖 Ejecución:")
    print("""
./yolo_tensorrt_detector \\
    ../models/model.fp16.engine \\
    ../scripts/videos/prueba2.mp4 \\
    0.5
    """)
    
    print("\n📊 Output esperado:")
    print("""
📂 Cargando engine TensorRT: ../models/model.fp16.engine
✅ Engine cargado
   Input: 2560000 elementos
   Output: 25200 elementos

⏱️  Tiempo inferencia: 15 ms

📊 RESULTADOS
======================================================================
Frames procesados: 300
Rostros detectados: 542
FPS promedio: 66.67
Latencia promedio: 15.00 ms/frame
    """)

def example_7_complete_pipeline():
    """Ejemplo 7: Pipeline completo (inicio a fin)."""
    print("\n" + "=" * 70)
    print("EJEMPLO 7: Pipeline Completo (Inicio a Fin)")
    print("=" * 70)
    
    print("\n📋 Checklist de ejecución:")
    print("""
1. Setup (5 min)
   ├─ .\\setup_hu04.ps1
   └─ python scripts/check_hu04_setup.py

2. Exportación ONNX (2 horas)
   └─ python scripts/export_to_tensorrt.py --model model.pt --output models --benchmark

3. Benchmarking (30 min)
   └─ python models/benchmark_onnx_vs_pytorch.py

4. Validación mAP (1 hora)
   └─ python scripts/validate_models.py

5. TensorRT (opcional, 3 horas)
   ├─ Descargar TensorRT desde https://developer.nvidia.com/tensorrt
   ├─ Instalar y configurar PATH
   └─ python models/convert_to_tensorrt.py

6. C++ + CUDA (opcional, 2-4 horas)
   ├─ cd scripts && mkdir build && cd build
   ├─ cmake .. -DTENSORRT_ROOT=/path/to/tensorrt
   ├─ cmake --build . --config Release
   └─ ./yolo_tensorrt_detector models/model.fp16.engine videos/prueba2.mp4

7. Reporte (1 hora)
   └─ Documentar todos los benchmarks y resultados
    """)

def main():
    print("\n" + "=" * 70)
    print("🎓 EJEMPLOS PRÁCTICOS - HU-04: OPTIMIZACIÓN DE INFERENCIA")
    print("=" * 70)
    
    # Listar ejemplos
    examples = [
        ("1", "Cargar modelo PyTorch", example_1_load_pytorch_model),
        ("2", "Exportar a ONNX", example_2_export_to_onnx),
        ("3", "Benchmarking", example_3_benchmark),
        ("4", "Validación Precisión", example_4_validate_precision),
        ("5", "Exportar a TensorRT", example_5_tensorrt_export),
        ("6", "Pipeline C++ TensorRT", example_6_cpp_tensorrt),
        ("7", "Pipeline Completo", example_7_complete_pipeline),
    ]
    
    print("\n📚 Ejemplos disponibles:\n")
    for num, title, _ in examples:
        print(f"  {num}. {title}")
    
    print("\n" + "=" * 70)
    
    # Mostrar todos los ejemplos
    for num, title, func in examples:
        func()
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📖 RESUMEN")
    print("=" * 70)
    
    print("""
PyTorch  ──(export)──>  ONNX  ──(convert)──>  TensorRT
├─ 45 ms/frame           ├─ 18 ms/frame        └─ 15 ms/frame
├─ Baseline              ├─ 2.5x speedup       ├─ 3x speedup
└─ mAP: 100%             ├─ 1.5% loss          └─ <2% loss
                         └─ GPU/CPU            └─ Solo GPU Nvidia

Recomendación:
├─ Desarrollo: PyTorch
├─ Producción (CPU): ONNX (1.5-2x rápido)
└─ Producción (GPU): TensorRT (2-5x rápido)
    """)
    
    print("=" * 70)
    print("\n✅ Ejecuta los scripts en este orden:")
    print("   1. python scripts/export_to_tensorrt.py")
    print("   2. python models/benchmark_onnx_vs_pytorch.py")
    print("   3. python scripts/validate_models.py")
    print("\n💡 Ver documentación completa: HU-04-OPTIMIZACION.md")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
