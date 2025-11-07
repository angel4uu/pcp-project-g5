"""
Setup rápido para HU-04: Optimización de Inferencia
Instala dependencias y ejecuta pasos iniciales automáticamente.
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, description=""):
    """Ejecutar comando y reportar resultado."""
    if description:
        print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description} falló")
        return False

def main():
    print("\n" + "=" * 70)
    print("🚀 SETUP HU-04: OPTIMIZACIÓN DE INFERENCIA")
    print("=" * 70)
    
    system = platform.system()
    
    # Paso 1: Crear venv
    print("\n[1/4] Preparando entorno virtual...")
    if not os.path.exists(".venv"):
        run_command(f"{sys.executable} -m venv .venv", 
                   "Crear entorno virtual")
    
    # Determinar comando activación
    if system == "Windows":
        activate = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\pip"
    else:
        activate = "source .venv/bin/activate"
        pip_cmd = ".venv/bin/pip"
    
    # Paso 2: Instalar dependencias
    print("\n[2/4] Instalando dependencias...")
    
    deps = [
        "ultralytics",
        "opencv-python",
        "opencv-contrib-python",
        "numpy",
        "torch>=2.0",
        "onnx",
        "onnxruntime",
    ]
    
    cmd = f"{pip_cmd} install " + " ".join(deps)
    run_command(cmd, "Instalar paquetes Python")
    
    # Paso 3: Crear directorio de modelos
    print("\n[3/4] Preparando directorios...")
    os.makedirs("models", exist_ok=True)
    print("✅ Directorio 'models' listo")
    
    # Paso 4: Exportar modelo
    print("\n[4/4] Exportando modelo YOLO a ONNX...")
    
    if os.path.exists("model.pt"):
        if system == "Windows":
            cmd = f"{pip_cmd} run python scripts/export_to_tensorrt.py --model model.pt --output models --benchmark"
        else:
            cmd = f"python scripts/export_to_tensorrt.py --model model.pt --output models --benchmark"
        
        run_command(cmd, "Exportar YOLO a ONNX")
    else:
        print("⚠️  model.pt no encontrado - salta exportación")
    
    # Resumen final
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETADO")
    print("=" * 70)
    
    print("\n📋 Próximos pasos:")
    print("   1. Activar venv:")
    if system == "Windows":
        print("      .venv\\Scripts\\activate")
    else:
        print("      source .venv/bin/activate")
    
    print("\n   2. Ejecutar benchmark:")
    print("      python models/benchmark_onnx_vs_pytorch.py")
    
    print("\n   3. Instalar TensorRT (opcional):")
    print("      Descargar de: https://developer.nvidia.com/tensorrt")
    
    print("\n   4. Compilar pipeline C++ (opcional):")
    print("      cd scripts && mkdir build && cd build")
    print("      cmake .. -DTENSORRT_ROOT=/ruta/a/tensorrt")
    print("      cmake --build . --config Release")
    
    print("\n📖 Ver documentación completa:")
    print("   cat HU-04-OPTIMIZACION.md")
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
