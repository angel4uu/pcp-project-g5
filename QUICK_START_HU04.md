# 🟡 HU-04: Optimización de Inferencia - Guía Rápida para Equipo 2

> **Este documento es para el Equipo 2 (2 personas) que trabaja en optimización con TensorRT/ONNX**

## 🎯 Objetivo de la Historia de Usuario

Convertir el modelo YOLO de PyTorch a formatos optimizados (ONNX, TensorRT) para lograr **2-5x speedup** sin pérdida de precisión (mAP).

---

## 🚀 Arranque en 5 Minutos (Windows)

### Paso 1: Abrir PowerShell y ejecutar

```powershell
# Navegar a la carpeta del proyecto
cd "C:\Users\USUARIO\Documents\Proyectos\Construccion-software\pcp-project-g5"

# Ejecutar script de setup (instala todo automáticamente)
.\setup_hu04.ps1
```

**Output esperado**:
```
================================
🚀 SETUP HU-04: WINDOWS POWERSHELL
================================

[0/4] Verificando Python...
  ✅ Python 3.11.x

[1/4] Preparando entorno virtual...
  ✅ Entorno virtual creado

[2/4] Actualizando pip...
  ✅ pip actualizado

[3/4] Instalando dependencias HU-04...
  ✅ (múltiples paquetes instalándose...)

[4/4] Verificando instalación...
  ✅ SETUP CORRECTO - ¡Listo para HU-04!

📋 Próximos pasos:
1️⃣  Exportar modelo...
...
```

---

## 📊 Flujo de Trabajo Típico (2 personas, 1 semana)

```
DÍA 1-2: Exportación
├─ Tarea 1: Exportar YOLO → ONNX
│  └─ Responsable: Persona A
│  └─ Comando: python scripts/export_to_tensorrt.py --model model.pt --output models --benchmark
│  └─ Validar: archivo models/model.onnx existe
│
├─ Tarea 2: Benchmarking Python
│  └─ Responsable: Persona B
│  └─ Comando: python models/benchmark_onnx_vs_pytorch.py
│  └─ Resultado: speedup 1.5-3x esperado
│
└─ Reunión: Revisar speedup vs PyTorch

DÍA 3-4: TensorRT (opcional, si GPU disponible)
├─ Tarea 3: Instalar TensorRT
│  └─ Responsable: Persona A
│  └─ URL: https://developer.nvidia.com/tensorrt
│  └─ Validar: trtexec --help
│
├─ Tarea 4: Exportar ONNX → TensorRT
│  └─ Responsable: Persona B
│  └─ Comando: python models/convert_to_tensorrt.py
│  └─ Resultado: models/model.fp16.engine
│
└─ Reunión: Validar engine TensorRT

DÍA 5-6: Validación
├─ Tarea 5: Validar mAP y precisión
│  └─ Responsable: Ambos
│  └─ Comando: python scripts/validate_models.py
│  └─ Criterios: mAP loss < 1%, similitud > 98%
│
└─ Tarea 6: Documentar resultados
   └─ Crear reporte final con benchmarks

DÍA 7: C++ + CUDA (avanzado, si tiempo disponible)
└─ Compilar pipeline C++ con TensorRT
   └─ Comando: cd scripts && mkdir build && cd build && cmake .. && make
   └─ Ejecutar: ./yolo_tensorrt_detector model.fp16.engine videos/prueba2.mp4
```

---

## 💻 Comandos Principales

### Exportar YOLO a ONNX

```powershell
# Activar entorno (si no está activo)
.\.venv\Scripts\Activate.ps1

# Exportar modelo
python scripts/export_to_tensorrt.py `
    --model model.pt `
    --output models `
    --benchmark
```

**Archivos generados**:
- `models/model.onnx` (modelo compilado)
- `models/benchmark_onnx_vs_pytorch.py` (script para benchmark)

---

### Benchmarking (medir velocidad)

```powershell
# Ejecutar benchmark de PyTorch vs ONNX
python models/benchmark_onnx_vs_pytorch.py
```

**Output esperado**:
```
============================================================
BENCHMARK: PyTorch vs ONNX Runtime
============================================================

⏱️  Benchmark PyTorch (GPU)
  Latencia: 45.32 ± 2.15 ms
  FPS: 22.07

⏱️  Benchmark ONNX Runtime (CUDA)
  Latencia: 18.45 ± 1.50 ms
  FPS: 54.20

📊 SPEEDUP ONNX: 2.46x
   PyTorch: 45.32 ms (22.07 FPS)
   ONNX:    18.45 ms (54.20 FPS)
```

---

### Validar Precisión

```powershell
# Comparar outputs PyTorch vs ONNX
python scripts/validate_models.py `
    --pytorch model.pt `
    --onnx models/model.onnx `
    --images scripts/images
```

**Criterios de aceptación**:
- ✅ Similitud ≥ 98% (outputs casi idénticos)
- ✅ mAP loss < 1% (precisión no decrece significativamente)

---

## 📂 Estructura de Archivos Generados

```
pcp-project-g5/
├── models/                          # Directorio de exportación
│   ├── model.onnx                  # ✅ Modelo ONNX compilado
│   ├── model.fp16.engine           # (generado por TensorRT)
│   ├── model.fp32.engine           # (alternativa precisión)
│   ├── benchmark_onnx_vs_pytorch.py    # (auto-generado)
│   └── convert_to_tensorrt.py      # (auto-generado)
├── scripts/
│   ├── export_to_tensorrt.py       # 🔄 Script principal exportación
│   ├── validate_models.py          # 🔄 Validador
│   ├── check_hu04_setup.py         # 🧪 Smoke test
│   ├── yolo_tensorrt_detector.cpp  # C++ pipeline
│   └── CMakeLists.txt              # Build system
├── HU-04-OPTIMIZACION.md           # 📖 Documentación completa
└── requirements-hu04.txt           # 📋 Dependencias
```

---

## 🔍 Troubleshooting

### Error: "No module named 'cv2'"
```powershell
pip install opencv-contrib-python
```

### Error: "No module named 'ultralytics'"
```powershell
pip install ultralytics
```

### Error: "CUDA not available"
```powershell
# Es normal, funcionará con CPU (más lento)
# Para habilitar CUDA:
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error: "trtexec not found"
```powershell
# Descargar TensorRT: https://developer.nvidia.com/tensorrt
# Añadir a PATH:
$env:PATH += ";C:\Program Files\TensorRT\bin"
```

---

## 📊 Métricas a Recolectar

Durante la ejecución, documentar:

| Métrica | PyTorch | ONNX | TensorRT |
|---------|---------|------|----------|
| FPS | ? | ? | ? |
| Latencia (ms) | ? | ? | ? |
| mAP@0.5 | ? | ? | ? |
| Memory (MB) | ? | ? | ? |
| Tamaño modelo | ? | ? | ? |

**Template de reporte**:
```
FECHA: 2025-11-06
RESPONSABLE: Persona A + Persona B

RESULTADOS:
- FPS ONNX vs PyTorch: 2.46x speedup ✓
- Similitud outputs: 98.5% ✓
- mAP loss: 0.3% ✓
- Status: LISTO PARA PRODUCCIÓN ✓
```

---

## 🎓 Conceptos Clave

### ONNX (Open Neural Network Exchange)
- Formato neutral para redes neuronales
- Interoperable (PyTorch → ONNX → TensorRT, ONNX.js, etc.)
- Típicamente 1-2x más rápido que PyTorch puro

### TensorRT
- Optimizador de Nvidia para CUDA/GPU
- Compilación JIT de modelos
- Típicamente 2-5x más rápido que ONNX
- Requiere GPU Nvidia

### mAP (mean Average Precision)
- Métrica de precisión para detección de objetos
- Rango: 0-100%
- Aceptable pérdida: < 1%

---

## 📞 Contacto y Ayuda

**Preguntas sobre exportación**:
- Ver: `scripts/export_to_tensorrt.py` (comentarios en código)
- Ejecutar: `python scripts/export_to_tensorrt.py --help`

**Preguntas sobre validación**:
- Ver: `scripts/validate_models.py`
- Ejecutar: `python scripts/validate_models.py --help`

**Preguntas sobre compilación C++**:
- Ver: `scripts/CMakeLists.txt`
- Ver: `HU-04-OPTIMIZACION.md` (sección C++)

**Documentación completa**:
```powershell
cat HU-04-OPTIMIZACION.md
```

---

## ✅ Checklist Final

- [ ] Setup completado (`.\setup_hu04.ps1` sin errores)
- [ ] Smoke test pasado (`python scripts/check_hu04_setup.py`)
- [ ] Modelo ONNX exportado (`models/model.onnx` existe)
- [ ] Benchmark ejecutado (speedup documentado)
- [ ] Validación mAP completada (criterios cumplidos)
- [ ] Reporte final generado
- [ ] Documentación actualizada

---

**Última actualización**: 2025-11-06  
**Responsables**: Equipo 2 (Optimización)  
**Estado**: 🟡 En Progreso
