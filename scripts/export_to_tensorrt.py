"""
Script para exportar modelos YOLO a ONNX y TensorRT.
HU-04: Optimización de Inferencia
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import onnx
import onnxruntime as ort

def check_cuda():
    """Verificar disponibilidad de CUDA"""
    cuda_available = torch.cuda.is_available()
    print(f"🔍 CUDA disponible: {cuda_available}")
    if cuda_available:
        print(f"   Dispositivo: {torch.cuda.get_device_name(0)}")
        print(f"   Versión CUDA: {torch.version.cuda}")
        print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
    return cuda_available

def export_yolo_to_onnx(model_path, output_dir="models", opset=12):
    """
    Exportar modelo YOLO a ONNX.
    
    Args:
        model_path: ruta al modelo YOLO (.pt)
        output_dir: directorio de salida
        opset: versión ONNX opset (default: 12)
    
    Returns:
        ruta al archivo ONNX exportado
    """
    print(f"\n📤 Exportando YOLO a ONNX desde: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Modelo no encontrado en {model_path}")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        model = YOLO(model_path)
        
        # Exportar a ONNX
        onnx_path = os.path.join(output_dir, Path(model_path).stem + ".onnx")
        print(f"   Exportando a: {onnx_path}")
        
        # Ultralytics export method
        exported_model = model.export(
            format='onnx',
            opset=opset,
            simplify=True,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"✅ ONNX exportado exitosamente")
        return str(exported_model)
        
    except Exception as e:
        print(f"❌ Error durante exportación a ONNX: {e}")
        return None

def validate_onnx_model(onnx_path, test_image_shape=(1, 3, 640, 640)):
    """
    Validar modelo ONNX cargándolo y ejecutando una inferencia dummy.
    
    Args:
        onnx_path: ruta al archivo ONNX
        test_image_shape: forma del tensor de entrada
    
    Returns:
        True si es válido, False en caso contrario
    """
    print(f"\n✔️  Validando modelo ONNX: {onnx_path}")
    
    try:
        # Cargar y validar el modelo ONNX
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("   ✓ Estructura ONNX válida")
        
        # Crear sesión ONNX Runtime
        session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print(f"   ✓ Sesión ONNX Runtime creada")
        
        # Inferencia dummy
        dummy_input = np.random.randn(*test_image_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: dummy_input})
        print(f"   ✓ Inferencia dummy exitosa (output shapes: {[o.shape for o in output]})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validando ONNX: {e}")
        return False

def export_to_tensorrt(onnx_path, output_dir="models", fp16=True, workspace_size=1):
    """
    Exportar modelo ONNX a TensorRT.
    Nota: Requiere TensorRT instalado y compilación en C++.
    Aquí generamos un script para hacerlo externamente.
    
    Args:
        onnx_path: ruta al archivo ONNX
        output_dir: directorio de salida
        fp16: usar precisión float16 (faster, lower memory)
        workspace_size: tamaño workspace en GB
    
    Returns:
        ruta al script que genera el engine TensorRT
    """
    print(f"\n📤 Preparando conversión a TensorRT: {onnx_path}")
    
    if not os.path.exists(onnx_path):
        print(f"❌ Error: ONNX no encontrado")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear script Python para convertir con trtexec o similar
    engine_name = Path(onnx_path).stem + (".fp16" if fp16 else ".fp32") + ".engine"
    engine_path = os.path.join(output_dir, engine_name)
    
    convert_script = f"""#!/usr/bin/env python3
\"\"\"Script para convertir ONNX a TensorRT usando trtexec.\"\"\"
import os
import subprocess
import sys

ONNX_PATH = r'{onnx_path}'
ENGINE_PATH = r'{engine_path}'
WORKSPACE_SIZE = {workspace_size * 1024}  # MB
FP16 = {str(fp16).lower()}

# Usar trtexec si está disponible
try:
    cmd = [
        'trtexec',
        f'--onnx={{ONNX_PATH}}',
        f'--saveEngine={{ENGINE_PATH}}',
        f'--workspace={{WORKSPACE_SIZE}}',
    ]
    
    if FP16:
        cmd.append('--fp16')
    
    print(f"Ejecutando: {{' '.join(cmd)}}")
    result = subprocess.run(cmd, check=True)
    
    if os.path.exists(ENGINE_PATH):
        print(f"✅ Engine TensorRT creado: {{ENGINE_PATH}}")
        sys.exit(0)
    else:
        print(f"❌ Error: Engine no creado")
        sys.exit(1)
        
except FileNotFoundError:
    print("❌ trtexec no encontrado. Instala TensorRT en el sistema.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {{e}}")
    sys.exit(1)
"""
    
    script_path = os.path.join(output_dir, "convert_to_tensorrt.py")
    with open(script_path, 'w') as f:
        f.write(convert_script)
    
    print(f"✅ Script generado: {script_path}")
    print(f"   Ejecuta: python {script_path}")
    print(f"   Generará: {engine_path}")
    
    return script_path

def create_inference_benchmark_script(onnx_path, output_dir="models"):
    """
    Crear script Python para benchmarking de velocidad ONNX vs PyTorch.
    """
    benchmark_script = f"""#!/usr/bin/env python3
\"\"\"Benchmark: YOLO PyTorch vs ONNX Runtime\"\"\"

import numpy as np
import time
import torch
import onnxruntime as ort
from ultralytics import YOLO
import cv2

def benchmark_pytorch(model_path, num_iterations=100):
    \"\"\"Benchmark de modelo YOLO PyTorch\"\"\"
    print("\\n⏱️  Benchmark PyTorch (GPU)")
    
    model = YOLO(model_path)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    dummy_input_torch = torch.from_numpy(dummy_input)
    if torch.cuda.is_available():
        dummy_input_torch = dummy_input_torch.cuda()
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input_torch)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Benchmark
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        _ = model(dummy_input_torch)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / avg_time
    
    print(f"  Latencia: {{avg_time:.2f}} ± {{std_time:.2f}} ms")
    print(f"  FPS: {{fps:.2f}}")
    return avg_time, fps

def benchmark_onnx(onnx_path, num_iterations=100):
    \"\"\"Benchmark de modelo ONNX Runtime con CUDA\"\"\"
    print("\\n⏱️  Benchmark ONNX Runtime (CUDA)")
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {{input_name: dummy_input}})
    
    # Benchmark
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        _ = session.run(None, {{input_name: dummy_input}})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / avg_time
    
    print(f"  Latencia: {{avg_time:.2f}} ± {{std_time:.2f}} ms")
    print(f"  FPS: {{fps:.2f}}")
    return avg_time, fps

if __name__ == "__main__":
    model_path = "../model.pt"  # Ajusta según tu setup
    onnx_path = r'{onnx_path}'
    
    print("=" * 60)
    print("BENCHMARK: PyTorch vs ONNX Runtime")
    print("=" * 60)
    
    pt_time, pt_fps = benchmark_pytorch(model_path)
    onnx_time, onnx_fps = benchmark_onnx(onnx_path)
    
    speedup = pt_time / onnx_time
    print(f"\\n📊 SPEEDUP ONNX: {{speedup:.2f}}x")
    print(f"   PyTorch: {{pt_time:.2f}} ms ({{pt_fps:.2f}} FPS)")
    print(f"   ONNX:    {{onnx_time:.2f}} ms ({{onnx_fps:.2f}} FPS)")
"""
    
    script_path = os.path.join(output_dir, "benchmark_onnx_vs_pytorch.py")
    with open(script_path, 'w') as f:
        f.write(benchmark_script)
    
    print(f"\n✅ Script benchmark creado: {script_path}")
    return script_path

def main():
    parser = argparse.ArgumentParser(
        description="Exportar modelo YOLO a ONNX y TensorRT para HU-04"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='../model.pt',
        help='Ruta al modelo YOLO (.pt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Directorio de salida'
    )
    parser.add_argument(
        '--tensorrt',
        action='store_true',
        help='Generar script para TensorRT'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Crear script de benchmarking'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=12,
        help='Versión ONNX opset'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 EXPORTADOR YOLO → ONNX → TENSORRT (HU-04)")
    print("=" * 70)
    
    # Verificar CUDA
    check_cuda()
    
    # Exportar a ONNX
    onnx_path = export_yolo_to_onnx(args.model, args.output, args.opset)
    
    if onnx_path:
        # Validar ONNX
        if validate_onnx_model(onnx_path):
            print("\n✅ Modelo ONNX válido y funcional")
            
            # Generar scripts adicionales
            if args.tensorrt:
                tensorrt_script = export_to_tensorrt(onnx_path, args.output)
                print(f"\n💡 Próximo paso TensorRT:")
                print(f"   python {tensorrt_script}")
            
            if args.benchmark:
                benchmark_script = create_inference_benchmark_script(onnx_path, args.output)
                print(f"\n💡 Para benchmarking:")
                print(f"   python {benchmark_script}")
            
            print("\n" + "=" * 70)
            print("✅ Exportación completada")
            print("=" * 70)
        else:
            print("\n❌ Validación ONNX fallida")
    else:
        print("\n❌ Exportación a ONNX fallida")

if __name__ == "__main__":
    main()
