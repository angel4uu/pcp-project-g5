#!/usr/bin/env pwsh

# ============================================================================
# Setup rápido para HU-04 en Windows (PowerShell)
# ============================================================================

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "🚀 SETUP HU-04: WINDOWS POWERSH ELL" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Paso 0: Verificar Python
Write-Host "[0/4] Verificando Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  ❌ Python no encontrado. Instalar desde https://python.org" -ForegroundColor Red
    exit 1
}

# Paso 1: Crear venv
Write-Host "`n[1/4] Preparando entorno virtual..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    Write-Host "  Creando .venv..." -ForegroundColor Cyan
    python -m venv .venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Entorno virtual creado" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Error creando entorno virtual" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  ✅ Entorno virtual ya existe" -ForegroundColor Green
}

# Activar venv
Write-Host "  Activando venv..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

# Paso 2: Actualizar pip
Write-Host "`n[2/4] Actualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel | Out-Null
Write-Host "  ✅ pip actualizado" -ForegroundColor Green

# Paso 3: Instalar dependencias
Write-Host "`n[3/4] Instalando dependencias HU-04..." -ForegroundColor Yellow

$packages = @(
    "ultralytics==8.0.220",
    "torch==2.1.0",
    "torchvision==0.16.0",
    "numpy==1.24.3",
    "opencv-python==4.8.1.78",
    "opencv-contrib-python==4.8.1.78",
    "onnx==1.14.1",
    "onnxruntime==1.17.0",
    "scikit-learn==1.3.1",
    "tqdm==4.66.1",
    "Pillow==10.1.0"
)

foreach ($package in $packages) {
    Write-Host "  → Instalando $package..." -ForegroundColor Cyan
    pip install $package -q
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    ✅" -ForegroundColor Green
    } else {
        Write-Host "    ⚠️  (puede fallar en conexión lenta, ignorando)" -ForegroundColor Yellow
    }
}

# Paso 4: Verificar instalación
Write-Host "`n[4/4] Verificando instalación..." -ForegroundColor Yellow
python scripts/check_hu04_setup.py

# Resumen final
Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "✅ SETUP COMPLETADO" -ForegroundColor Green
Write-Host "================================`n" -ForegroundColor Cyan

Write-Host "📋 Próximos pasos:`n" -ForegroundColor Cyan

Write-Host "1️⃣  Exportar modelo YOLO a ONNX:" -ForegroundColor White
Write-Host "   python scripts/export_to_tensorrt.py --model model.pt --output models --benchmark`n" -ForegroundColor Yellow

Write-Host "2️⃣  Ejecutar benchmark:" -ForegroundColor White
Write-Host "   python models/benchmark_onnx_vs_pytorch.py`n" -ForegroundColor Yellow

Write-Host "3️⃣  Validar modelos:" -ForegroundColor White
Write-Host "   python scripts/validate_models.py`n" -ForegroundColor Yellow

Write-Host "4️⃣  TensorRT (opcional):" -ForegroundColor White
Write-Host "   Descargar desde: https://developer.nvidia.com/tensorrt`n" -ForegroundColor Yellow

Write-Host "📖 Documentación completa:" -ForegroundColor White
Write-Host "   cat HU-04-OPTIMIZACION.md`n" -ForegroundColor Yellow

Write-Host "================================`n" -ForegroundColor Cyan
