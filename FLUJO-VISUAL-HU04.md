# 🎯 Flujo Visual - HU-04: Optimización de Inferencia

## Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PIPELINE DE OPTIMIZACIÓN HU-04                         │
└─────────────────────────────────────────────────────────────────────────────┘

                                   MODEL.PT
                                  (PyTorch)
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌──────────────┐  ┌─────────────┐  ┌────────────┐
            │   INFERENCE  │  │   INFERENCE │  │ INFERENCE  │
            │   PyTorch    │  │    ONNX     │  │ TensorRT   │
            │              │  │             │  │            │
            │ FPS: 22      │  │ FPS: 54     │  │ FPS: 67    │
            │ mAP: 92.5%   │  │ mAP: 92.3%  │  │ mAP: 92.1% │
            └──────┬───────┘  └──────┬──────┘  └────┬───────┘
                   │                 │              │
                   │            1.5-2x            2-5x
                   │            SPEEDUP          SPEEDUP
                   │                 │              │
        ┌──────────┴─────────────┬───┴──────────────┼──────────────┐
        │                        │                  │              │
    BASELINE                 PRODUCCIÓN          PRODUCCIÓN      EDGE
   (Desarrollo)              (CPU/GPU)         (GPU NVIDIA)    (Smartphone)
        │                        │                  │              │
        │                        ▼                  ▼              │
        │                   model.onnx          model.engine      │
        │                   (48 MB)             (15 MB)           │
        │                        │                  │              │
        └────────────────────────┼──────────────────┼──────────────┘
                                 │                  │
                       ┌─────────▼──────────┐  ┌──▼──────────────┐
                       │  VALIDATION (mAP)  │  │   C++ + CUDA    │
                       │                    │  │   + TensorRT    │
                       │ Loss: <1%          │  │                 │
                       │ Similarity: >98%   │  │ Compilado       │
                       └────────────────────┘  │ Optimizado      │
                                               └─────────────────┘
```

---

## Flujo de Tareas (Gantt)

```
DÍA 1  DÍA 2  DÍA 3  DÍA 4  DÍA 5  DÍA 6  DÍA 7
│      │      │      │      │      │      │
├──────┼──────┼──────┼──────┼──────┼──────┼─
│
├─ Setup
│  ████ (4h)
│
├─ ONNX Exportación
│  ████████ (8h)
│
├─ Benchmarking Python
│  ████ (4h)
│
├─ Instalación TensorRT (Opcional)
│          ████ (4h)
│
├─ TensorRT Exportación (Opcional)
│              ████████ (8h)
│
├─ Compilación C++ (Opcional)
│                  ████ (4-8h)
│
├─ Validación mAP
│          ████ (4h)
│
└─ Reporte Final
                            ██ (2h)
```

---

## Decisión: ¿Cuál usar?

```
┌─────────────────────────────────────────────────────────────────┐
│                    MATRIZ DE DECISIÓN                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DESARROLLO:                                                    │
│  └─ PyTorch ✅ (código nativo, fácil debug)                     │
│                                                                 │
│  PRODUCCIÓN (CPU/GPU):                                          │
│  └─ ONNX ✅ (1.5-2x rápido, cross-platform)                    │
│                                                                 │
│  PRODUCCIÓN (GPU NVIDIA):                                       │
│  └─ TensorRT ✅ (2-5x rápido, máxima performance)              │
│                                                                 │
│  MOBILE/EDGE:                                                   │
│  └─ ONNX + TFLite (future work)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Flujo de Ejecución Comandos

```
START
  │
  ├─► Setup
  │   $ .\setup_hu04.ps1
  │   └─► Crea .venv, instala deps
  │
  ├─► Verificar
  │   $ python scripts/check_hu04_setup.py
  │   └─► ✅ Todo OK?
  │
  ├─► ONNX Export
  │   $ python scripts/export_to_tensorrt.py --model model.pt --output models --benchmark
  │   └─► Genera: models/model.onnx
  │
  ├─► Python Benchmark
  │   $ python models/benchmark_onnx_vs_pytorch.py
  │   └─► ✅ Speedup >= 1.5x?
  │
  ├─► Validación mAP
  │   $ python scripts/validate_models.py
  │   └─► ✅ mAP loss < 1%?
  │
  ├─┐ (OPCIONAL) TensorRT
  │ │
  │ ├─► Instalar TensorRT
  │ │   $ Descargar de https://developer.nvidia.com/tensorrt
  │ │
  │ ├─► ONNX → TensorRT
  │ │   $ python models/convert_to_tensorrt.py
  │ │   └─► Genera: models/model.fp16.engine
  │ │
  │ └─► TensorRT Benchmark
  │     $ ./yolo_tensorrt_detector model.engine video.mp4
  │     └─► ✅ Speedup >= 2.5x?
  │
  ├─► Reporte
  │   $ Documentar todos los benchmarks y mAP
  │
  └─► END

```

---

## Archivo → Responsabilidad → Output

```
Persona A (Export):              Persona B (Benchmark):
├─ export_to_tensorrt.py    ├─ validate_models.py
├─ yolo_tensorrt_detector.cpp│─ benchmark_onnx_vs_pytorch.py
├─ CMakeLists.txt           └─ check_hu04_setup.py
└─ setup_hu04.ps1
   │                           │
   ▼                           ▼
models/model.onnx         Reporte de speedup
models/model.engine       Validación mAP
yolo_tensorrt_detector    Métricas finales

REUNIÓN: Revisar speedup vs mAP
  └─► Decisión: ¿ONNX o TensorRT para producción?
```

---

## Métricas Esperadas

```
┌─────────────────────────────────────────────────────────────┐
│                   BENCHMARKS ESPERADOS                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PyTorch (Baseline):                                        │
│  ├─ FPS: 20-25                                              │
│  ├─ Latencia: 40-50 ms                                      │
│  ├─ mAP@0.5: 92.5%                                          │
│  └─ Status: Baseline (referencia)                           │
│                                                             │
│  ONNX (Interop):                                            │
│  ├─ FPS: 50-60 (1.5-2x speedup)                             │
│  ├─ Latencia: 18-25 ms                                      │
│  ├─ mAP@0.5: 92.3% (< 1% loss)                              │
│  └─ Status: ✅ ACEPTADO                                     │
│                                                             │
│  TensorRT (Optimizado):                                     │
│  ├─ FPS: 60-100 (2.5-5x speedup)                            │
│  ├─ Latencia: 10-18 ms                                      │
│  ├─ mAP@0.5: 92.1% (< 2% loss)                              │
│  └─ Status: ✅ ACEPTADO                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Archivos → Equipos → Timeline

```
┌─────────────────────────────────────────────────────────┐
│                  MATRIZ RESPONSABILIDAD                 │
├──────────────────────────────────────────────────────────┤
│ Archivo               │ Persona │ Días │ Dependencia   │
├──────────────────────────────────────────────────────────┤
│ setup_hu04.ps1        │ Setup   │ 1d   │ Ninguno       │
│ export_to_tensorrt.py │ A       │ 2d   │ Setup         │
│ validate_models.py    │ B       │ 2d   │ Setup + A     │
│ CMakeLists.txt        │ A       │ 1d   │ Setup + TRT   │
│ yolo_tensorrt_detector│ A       │ 2d   │ CMake + TRT   │
│ Reporte Final         │ A+B     │ 1d   │ Todas         │
├──────────────────────────────────────────────────────────┤
│ TOTAL ESTIMADO: 7 días (1 semana intenso)              │
└──────────────────────────────────────────────────────────┘
```

---

## Puntos de Decisión

```
┌─ SETUP OK?
│  └─► NO → Troubleshoot (ver QUICK_START_HU04.md)
│  └─► SÍ → Continuar
│
├─ ONNX SPEEDUP >= 1.5x?
│  └─► NO → Revisar export settings
│  └─► SÍ → Continuar
│
├─ mAP LOSS < 1%?
│  └─► NO → Revisar quantización
│  └─► SÍ → ONNX APROBADO ✅
│
├─ ¿TENEMOS GPU NVIDIA?
│  └─► NO → Parar aquí (ONNX es suficiente)
│  └─► SÍ → Continuar con TensorRT
│
├─ TENSORRT SPEEDUP >= 2.5x?
│  └─► NO → Revisar configuración GPU
│  └─► SÍ → Continuar
│
└─ mAP LOSS < 2%?
   └─► NO → Revisar cuantización FP16 vs INT8
   └─► SÍ → TENSORRT APROBADO ✅
```

---

## Escalabilidad Post-HU-04

```
FUTURO (HU-05+):
├─ Optimizaciones avanzadas
│  ├─ Quantización INT8 (más rápido, menos preciso)
│  ├─ Pruning (reducir pesos no necesarios)
│  └─ Knowledge Distillation (modelo más pequeño)
│
├─ Deployment
│  ├─ Docker container con TensorRT
│  ├─ API REST para inferencia
│  └─ Load balancing con GPU pool
│
└─ Monitoring
   ├─ Benchmarking en hardware diverso
   ├─ A/B testing PyTorch vs TensorRT
   └─ Métricas en tiempo real
```

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────┐
│              QUICK REFERENCE - HU-04                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  SETUP:                                                  │
│  $ .\setup_hu04.ps1                                      │
│                                                          │
│  EXPORT ONNX:                                            │
│  $ python scripts/export_to_tensorrt.py \               │
│    --model model.pt --output models --benchmark         │
│                                                          │
│  BENCHMARK:                                              │
│  $ python models/benchmark_onnx_vs_pytorch.py           │
│                                                          │
│  VALIDAR:                                                │
│  $ python scripts/validate_models.py                    │
│                                                          │
│  TENSORRT (si GPU):                                      │
│  $ python models/convert_to_tensorrt.py                 │
│                                                          │
│  COMPILAR C++:                                           │
│  $ cd scripts && mkdir build && cd build                │
│  $ cmake .. && make                                      │
│                                                          │
│  EJECUTAR C++:                                           │
│  $ ./yolo_tensorrt_detector model.engine video.mp4      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

**Última actualización**: 2025-11-06
