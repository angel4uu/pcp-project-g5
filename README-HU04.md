# 📚 ÍNDICE MAESTRO - HU-04: Optimización de Inferencia

## 🎯 ¿Por dónde empiezo?

**Eres nuevo en el proyecto?**  
→ Lee esto primero: **`ENTREGA-HU04.md`** (10 min)

**¿Tienes prisa?**  
→ Salta a: **`QUICK_START_HU04.md`** (ejecuta `.\setup_hu04.ps1` y comienza)

**¿Necesitas todo de una?**  
→ Documentación técnica: **`HU-04-OPTIMIZACION.md`**

---

## 📖 Documentos (En Orden de Lectura)

### 1. ENTREGA-HU04.md ⭐ COMIENZA AQUÍ
**¿Qué es?** Resumen ejecutivo de todo lo entregado  
**Duración**: 10 minutos  
**Audiencia**: Todos  
**Contiene**:
- ✓ Qué se entregó (14 archivos)
- ✓ Cómo empezar en 5 minutos
- ✓ Criterios de aceptación
- ✓ Dependencias
- ✓ Próximos pasos

---

### 2. QUICK_START_HU04.md ⭐ GUÍA OPERACIONAL
**¿Qué es?** Paso a paso para el Equipo 2  
**Duración**: 30 minutos  
**Audiencia**: Equipo 2 (2 personas)  
**Contiene**:
- ✓ Setup en 5 minutos
- ✓ Flujo de trabajo 1 semana
- ✓ Comandos principales
- ✓ Métricas a recolectar
- ✓ Troubleshooting

**👉 Comienza aquí después de leer ENTREGA-HU04.md**

---

### 3. HU-04-RESUMEN-EJECUTIVO.md
**¿Qué es?** Información para PMs y líderes  
**Duración**: 15 minutos  
**Audiencia**: Líderes, PMs, stakeholders  
**Contiene**:
- ✓ Status actual (🟡 En progreso)
- ✓ Timebox estimado
- ✓ Thresholds de aceptación
- ✓ Dashboard de progreso
- ✓ Riesgos y dependencias

---

### 4. HU-04-OPTIMIZACION.md 📘 REFERENCIA TÉCNICA
**¿Qué es?** Documentación técnica completa  
**Duración**: 45 minutos (lectura completa)  
**Audiencia**: Desarrolladores (ambas personas)  
**Contiene**:
- ✓ Tareas 1-4 detalladas
- ✓ Guía rápida paso a paso
- ✓ Validación de precisión (mAP)
- ✓ Archivos generados
- ✓ Thresholds de aceptación
- ✓ Troubleshooting

**👉 Referencia cuando tengas dudas técnicas**

---

### 5. EJEMPLOS_HU04.md
**¿Qué es?** Ejemplos prácticos de código  
**Duración**: 30 minutos  
**Audiencia**: Desarrolladores  
**Contiene**:
- ✓ 7 ejemplos paso a paso
- ✓ Código real con comentarios
- ✓ Output esperado para cada ejemplo
- ✓ Pipeline completo

**👉 Ejecuta los ejemplos mientras lees**

---

### 6. FLUJO-VISUAL-HU04.md
**¿Qué es?** Diagramas y flujos visuales  
**Duración**: 15 minutos  
**Audiencia**: Visual learners  
**Contiene**:
- ✓ Arquitectura visual
- ✓ Flujo de tareas (Gantt)
- ✓ Matriz de decisión
- ✓ Quick reference card

**👉 Consulta cuando necesites visualizar**

---

## 🐍 Scripts Python (En `scripts/`)

### export_to_tensorrt.py (400+ líneas)
**¿Qué hace?** Exporta YOLO → ONNX/TensorRT  
**Responsable**: Persona A (Export)  
**Tiempo**: 2 horas  
**Comando**:
```powershell
python scripts/export_to_tensorrt.py --model model.pt --output models --benchmark
```
**Resultado**: `models/model.onnx` (48 MB)

---

### validate_models.py (300+ líneas)
**¿Qué hace?** Valida precisión (mAP) y comparar outputs  
**Responsable**: Persona B (Validación)  
**Tiempo**: 1 hora  
**Comando**:
```powershell
python scripts/validate_models.py --pytorch model.pt --onnx models/model.onnx
```
**Resultado**: Reporte mAP, similitud outputs

---

### check_hu04_setup.py (150+ líneas)
**¿Qué hace?** Verifica que todo está correctamente instalado  
**Responsable**: Cualquiera (Setup check)  
**Tiempo**: 2 minutos  
**Comando**:
```powershell
python scripts/check_hu04_setup.py
```
**Resultado**: ✅ Setup OK o ❌ Problemas encontrados

---

### setup_hu04.py (100+ líneas)
**¿Qué hace?** Setup multiplataforma (Python)  
**Responsable**: Cualquiera (Initial setup)  
**Tiempo**: 10 minutos  
**Comando**:
```powershell
python scripts/setup_hu04.py
```
**Resultado**: Entorno virtual + dependencias instaladas

---

## 🔧 Scripts C++ (En `scripts/`)

### yolo_tensorrt_detector.cpp (450+ líneas)
**¿Qué hace?** Pipeline completo C++ + CUDA + TensorRT  
**Responsable**: Persona A (C++ avanzado, opcional)  
**Tiempo**: 2-4 horas (compilación + testing)  
**Requisitos**:
- TensorRT 8.6+ instalado
- CUDA 11.8+ instalado
- CMake 3.15+ instalado
- Visual Studio 2019+ (Windows)

**Compilar**:
```powershell
cd scripts
mkdir build
cd build
cmake .. -DTENSORRT_ROOT="C:\Program Files\TensorRT"
cmake --build . --config Release -j8
```

**Ejecutar**:
```powershell
cd Release
.\yolo_tensorrt_detector.exe ..\models\model.fp16.engine ..\scripts\videos\prueba2.mp4 0.5
```

**Resultado**: Ejecutable que procesa video en tiempo real con TensorRT

---

### CMakeLists.txt
**¿Qué hace?** Build system para compilar C++ + CUDA  
**Responsable**: Persona A (si compila C++)  
**Configuración**:
- OpenCV + CUDA + TensorRT linkadas
- Optimizaciones compilación (-O3)
- CUDA separable compilation habilitada

---

## ⚙️ Configuración

### setup_hu04.ps1 (PowerShell, Windows)
**¿Qué hace?** Setup automático completo (Windows)  
**Responsable**: Cualquiera (Initial setup)  
**Tiempo**: 15 minutos  
**Pasos**:
1. Verifica Python
2. Crea entorno virtual (.venv)
3. Instala todas las dependencias
4. Verifica setup
5. Imprime próximos pasos

**Ejecutar**:
```powershell
.\setup_hu04.ps1
```

---

### requirements-hu04.txt
**¿Qué es?** Lista de dependencias exactas  
**¿Cuándo usarlo?** Si `setup_hu04.ps1` falla  
**Usar manualmente**:
```powershell
pip install -r requirements-hu04.txt
```

---

## 📊 Flujo de Trabajo (Cómo Usar Todo)

### Día 1: Setup + ONNX Export

```
Mañana:
├─ 09:00 → Leer ENTREGA-HU04.md (10 min)
├─ 09:10 → Leer QUICK_START_HU04.md (20 min)
├─ 09:30 → Ejecutar .\setup_hu04.ps1 (15 min)
├─ 09:45 → Ejecutar python scripts/check_hu04_setup.py (2 min)
└─ 09:50 → ✅ Setup completado

Tarde:
├─ 14:00 → PERSONA A: Exportar ONNX
│   $ python scripts/export_to_tensorrt.py --model model.pt --output models --benchmark
│   (~ 2 horas)
└─ 16:00 → Archivo models/model.onnx generado ✅
```

---

### Día 2: Benchmarking + Validación

```
Mañana:
├─ 09:00 → PERSONA B: Benchmarking
│   $ python models/benchmark_onnx_vs_pytorch.py
│   (~ 30 min)
├─ 09:30 → PERSONA B: Validación mAP
│   $ python scripts/validate_models.py
│   (~ 1 hora)
└─ 10:30 → ✅ Métricas ONNX recolectadas

Tarde:
└─ 14:00 → REUNIÓN: Revisar speedup vs PyTorch
           ├─ Speedup >= 1.5x? → ✅ ONNX APROBADO
           ├─ mAP loss < 1%? → ✅ ONNX APROBADO
           └─ ¿Continuar con TensorRT? → Si tenemos GPU
```

---

### Día 3-4: TensorRT (Opcional, si GPU disponible)

```
Si tenemos GPU NVIDIA:
├─ Descargar TensorRT desde https://developer.nvidia.com/tensorrt
├─ Instalar y configurar PATH
├─ Ejecutar python models/convert_to_tensorrt.py
├─ Resultado: models/model.fp16.engine
└─ Benchmarking y validación nuevamente
```

---

### Día 5: C++ + CUDA (Avanzado, Optional)

```
Si queremos máximo rendimiento:
├─ cd scripts && mkdir build && cd build
├─ cmake .. -DTENSORRT_ROOT=/path/to/tensorrt
├─ cmake --build . --config Release -j8
├─ ./yolo_tensorrt_detector model.engine video.mp4
└─ Medir FPS y latencia con C++
```

---

### Día 6-7: Reporte Final

```
├─ Documentar todos los benchmarks en una tabla
├─ Documentar mAP para cada modelo
├─ Crear gráficos (opcional): speedup, mAP
├─ Conclusiones: ¿Qué modelo usar en producción?
└─ Presentar a stakeholders
```

---

## 🎓 Conceptos Clave

| Concepto | Explicación | Relevancia |
|----------|------------|-----------|
| **PyTorch** | Framework ML, forma nativa de escribir modelos | Baseline, desarrollo |
| **ONNX** | Formato estándar cross-platform | Export, interop |
| **TensorRT** | Optimizador Nvidia para GPUs | Max speedup (GPU only) |
| **Benchmarking** | Medir velocidad (FPS, latencia) | Validar speedup |
| **mAP** | Métrica de precisión (0-100%) | Validar que no pierde accuracy |
| **Quantización** | Reducir precisión (FP32→FP16) | Trade-off speedup vs precision |
| **Inferencia** | Hacer predicciones (opuesto a training) | Lo que hacemos aquí |
| **CUDA** | Programación GPU Nvidia | Acelerar computación |

---

## 🚨 Si Algo Sale Mal

| Problema | Solución |
|----------|----------|
| "No module named cv2" | `pip install opencv-contrib-python` |
| "No module named ultralytics" | `pip install ultralytics` |
| "CUDA not available" | Normal en CPU. Ver sección GPU en QUICK_START_HU04.md |
| "trtexec not found" | Instalar TensorRT e añadir a PATH |
| "Exportación falla" | Ver comentarios en export_to_tensorrt.py |
| "mAP loss > 2%" | Revisar quantización, usar FP32 en lugar de FP16 |

→ **Ver QUICK_START_HU04.md sección Troubleshooting para más detalles**

---

## ✅ Checklist Semanal

```
LUNES
├─ [ ] Leer ENTREGA-HU04.md
├─ [ ] Leer QUICK_START_HU04.md
├─ [ ] Ejecutar .\setup_hu04.ps1
├─ [ ] Ejecutar check_hu04_setup.py
└─ [ ] Exportar ONNX (Persona A)

MARTES
├─ [ ] Benchmarking (Persona B)
├─ [ ] Validación mAP (Persona B)
├─ [ ] Reunión: Revisar métricas
└─ [ ] Decisión: ¿TensorRT?

MIÉRCOLES-JUEVES (Optional TensorRT)
├─ [ ] Instalar TensorRT
├─ [ ] Convertir ONNX → TensorRT
├─ [ ] Benchmarking TensorRT
└─ [ ] Validación mAP TensorRT

VIERNES (Optional C++)
├─ [ ] Compilar C++ + CUDA
├─ [ ] Testing ejecutable
└─ [ ] Métricas C++

PRÓXIMA SEMANA
├─ [ ] Reporte final
├─ [ ] Presentación resultados
└─ [ ] Deployment (HU-05)
```

---

## 🎯 Criterios de Éxito

Habrás completado HU-04 cuando:

- ✅ ONNX exportado y funcional
- ✅ Speedup ONNX >= 1.5x vs PyTorch
- ✅ mAP loss ONNX < 1%
- ✅ Reporte con todas las métricas generado
- ✅ (Opcional) TensorRT exportado
- ✅ (Opcional) Speedup TensorRT >= 2.5x
- ✅ (Opcional) mAP loss TensorRT < 2%
- ✅ (Avanzado) C++ compilado y funcionando

---

## 📞 Soporte Rápido

**Pregunta** → **Documentación**

| Q | Documento |
|---|-----------|
| ¿Cómo empiezo? | QUICK_START_HU04.md |
| ¿Qué es ONNX/TensorRT? | HU-04-RESUMEN-EJECUTIVO.md |
| ¿Tengo error? | QUICK_START_HU04.md → Troubleshooting |
| ¿Cómo compilo C++? | HU-04-OPTIMIZACION.md → Paso 6 |
| ¿Qué código ejecuto? | EJEMPLOS_HU04.md |
| ¿Visualizar flujo? | FLUJO-VISUAL-HU04.md |
| ¿Ver todos los archivos? | ENTREGA-HU04.md |

---

## 🎁 Bonus: Scripts Auto-generados

Durante la exportación, estos scripts se generan automáticamente:

```
models/benchmark_onnx_vs_pytorch.py   ← Auto-generado
models/convert_to_tensorrt.py          ← Auto-generado
```

**No los edites**, se regeneran cada vez que exportas.

---

## 📈 Roadmap Post HU-04

Después de completar HU-04:

- **HU-05**: Deployment (Docker, API REST)
- **HU-06**: Monitoring (métricas en tiempo real)
- **HU-07**: Optimizaciones avanzadas (INT8 quantization, pruning)
- **HU-08**: Multi-GPU support
- **HU-09**: Mobile optimization (ONNX Lite, TFLite)

---

## 🎓 Referencias Externas

- [Ultralytics YOLO Export](https://docs.ultralytics.com/modes/export/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [CUDA Programming](https://docs.nvidia.com/cuda/)

---

## 📊 Resumen Visual

```
START
  │
  ├─► Leer ENTREGA-HU04.md (10 min)
  │
  ├─► Leer QUICK_START_HU04.md (20 min)
  │
  ├─► Ejecutar .\setup_hu04.ps1 (15 min)
  │
  ├─► ONNX Export (2 horas)
  │
  ├─► Benchmarking (1 hora)
  │
  ├─► Validación mAP (1 hora)
  │
  ├─┐ (Opcional) TensorRT
  │ ├─► Instalación (1 hora)
  │ ├─► Exportación (2 horas)
  │ └─► Benchmarking (1 hora)
  │
  ├─┐ (Avanzado) C++
  │ ├─► Compilación (1-2 horas)
  │ └─► Testing (1 hora)
  │
  ├─► Reporte Final (2 horas)
  │
  └─► END ✅ HU-04 COMPLETADA

Timeline: 1 semana (7 días)
Esfuerzo: 2 personas
Status: 🟡 Listo para empezar
```

---

**Última actualización**: 2025-11-06  
**Versión**: 1.0  
**Equipo**: 2 (Optimización)
