"""
Smoke Test - Verificar que todo esté correctamente configurado para HU-04
"""

import sys
import os

def check_imports():
    """Verificar que todos los imports críticos funcionan."""
    print("\n🔍 Verificando imports...")
    
    imports_required = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'ultralytics': 'Ultralytics YOLO',
        'onnx': 'ONNX',
        'onnxruntime': 'ONNX Runtime',
    }
    
    all_ok = True
    for module, name in imports_required.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            all_ok = False
    
    return all_ok

def check_files():
    """Verificar que los archivos necesarios existen."""
    print("\n📁 Verificando archivos...")
    
    files_required = {
        'scripts/export_to_tensorrt.py': 'Exportador ONNX/TensorRT',
        'scripts/yolo_tensorrt_detector.cpp': 'Pipeline C++ TensorRT',
        'scripts/CMakeLists.txt': 'Build system C++',
        'scripts/validate_models.py': 'Validador de modelos',
        'HU-04-OPTIMIZACION.md': 'Documentación HU-04',
    }
    
    all_ok = True
    for filepath, name in files_required.items():
        if os.path.exists(filepath):
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} ({filepath} no encontrado)")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Verificar disponibilidad de CUDA."""
    print("\n🔌 Verificando CUDA...")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"  ✅ CUDA disponible")
            print(f"     Dispositivo: {device_name}")
            print(f"     Versión: {cuda_version}")
        else:
            print(f"  ⚠️  CUDA no disponible (usarán CPU)")
        
        return cuda_available
    except Exception as e:
        print(f"  ❌ Error verificando CUDA: {e}")
        return False

def check_model():
    """Verificar que el modelo base existe."""
    print("\n🤖 Verificando modelo...")
    
    model_paths = [
        'model.pt',
        'scripts/models/best.pt',
        'yolov8n-widerface-v2/best.pt'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✅ Modelo encontrado: {path} ({size_mb:.1f} MB)")
            return True
    
    print(f"  ⚠️  Modelo no encontrado en ubicaciones estándar")
    print(f"     Ubicaciones esperadas: {', '.join(model_paths)}")
    return False

def check_directories():
    """Verificar estructura de directorios."""
    print("\n📂 Verificando directorios...")
    
    dirs = {
        'scripts': 'Scripts Python',
        'scripts/images': 'Imágenes de prueba',
        'scripts/videos': 'Videos de prueba',
        'models': 'Modelos exportados',
    }
    
    for dirname, name in dirs.items():
        if os.path.exists(dirname):
            print(f"  ✅ {name}")
        else:
            print(f"  ⚠️  {name} ({dirname} no existe, creando...)")
            os.makedirs(dirname, exist_ok=True)

def main():
    print("\n" + "=" * 70)
    print("🧪 SMOKE TEST - HU-04: OPTIMIZACIÓN DE INFERENCIA")
    print("=" * 70)
    
    results = {
        'imports': check_imports(),
        'files': check_files(),
        'cuda': check_cuda(),
        'model': check_model(),
    }
    
    check_directories()
    
    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPruebas pasadas: {passed}/{total}")
    
    if passed == total:
        print("\n✅ SETUP CORRECTO - ¡Listo para HU-04!")
        print("\nPróximos pasos:")
        print("  1. python scripts/export_to_tensorrt.py --model model.pt --output models --benchmark")
        print("  2. python models/benchmark_onnx_vs_pytorch.py")
        print("  3. Ver: HU-04-OPTIMIZACION.md")
        return 0
    else:
        print("\n⚠️  Algunas pruebas fallaron. Por favor, verificar logs arriba.")
        print("\nResolución:")
        if not results['imports']:
            print("  - Instalar: pip install -r requirements-hu04.txt")
        if not results['files']:
            print("  - Verificar que los scripts están presentes")
        if not results['cuda']:
            print("  - CUDA opcional. Script funcionará con CPU.")
        if not results['model']:
            print("  - Colocar modelo YOLO en: model.pt o scripts/models/best.pt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
