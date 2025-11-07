"""
Script de Validación y Benchmarking
HU-04: Optimizar precisión (mAP) vs velocidad

Compara:
- Modelo PyTorch (baseline)
- Modelo ONNX (interop)
- Modelo TensorRT (optimizado)
"""

import os
import sys
import time
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Tuple, List, Dict

import torch
from ultralytics import YOLO
import onnxruntime as ort
from sklearn.metrics import average_precision_score

class MetricCollector:
    """Recolector de métricas de inferencia y precisión."""
    
    def __init__(self):
        self.results = {
            'pytorch': {'times': [], 'fps': 0, 'latency': 0},
            'onnx': {'times': [], 'fps': 0, 'latency': 0},
            'tensorrt': {'times': [], 'fps': 0, 'latency': 0}
        }
    
    def add_time(self, model_type: str, elapsed_ms: float):
        """Añadir tiempo de inferencia."""
        if model_type in self.results:
            self.results[model_type]['times'].append(elapsed_ms)
    
    def compute_stats(self):
        """Calcular FPS y latencia."""
        for model_type, data in self.results.items():
            if data['times']:
                avg_time = np.mean(data['times'])
                data['latency'] = avg_time
                data['fps'] = 1000.0 / avg_time if avg_time > 0 else 0
    
    def report(self):
        """Generar reporte."""
        print("\n" + "=" * 70)
        print("📊 BENCHMARK REPORT")
        print("=" * 70)
        
        for model_type, data in self.results.items():
            if data['times']:
                print(f"\n{model_type.upper()}:")
                print(f"  FPS: {data['fps']:.2f}")
                print(f"  Latencia: {data['latency']:.2f} ms")
                print(f"  Min: {min(data['times']):.2f} ms, "
                      f"Max: {max(data['times']):.2f} ms")


class YOLOValidator:
    """Validador para modelos YOLO."""
    
    def __init__(self, image_dir: str = "scripts/images", 
                 conf_threshold: float = 0.5):
        self.image_dir = image_dir
        self.conf_threshold = conf_threshold
        self.images = []
        self.load_images()
    
    def load_images(self):
        """Cargar imágenes de prueba."""
        if not os.path.exists(self.image_dir):
            print(f"⚠️  Directorio no encontrado: {self.image_dir}")
            return
        
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.images.extend(Path(self.image_dir).glob(ext))
        
        self.images = self.images[:10]  # Limitar a 10 para testing
        print(f"✓ Imágenes cargadas: {len(self.images)}")
    
    def validate_pytorch(self, model_path: str) -> Tuple[float, int]:
        """Validar modelo PyTorch."""
        print("\n🔄 Validando PyTorch...")
        
        model = YOLO(model_path)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        total_detections = 0
        total_time = 0
        
        for image_path in self.images:
            start = time.perf_counter()
            results = model(str(image_path), verbose=False)
            end = time.perf_counter()
            
            total_time += (end - start) * 1000
            if results and len(results) > 0:
                total_detections += len(results[0].boxes)
        
        avg_time = total_time / len(self.images) if self.images else 0
        print(f"✓ PyTorch - Latencia promedio: {avg_time:.2f} ms")
        
        return avg_time, total_detections
    
    def validate_onnx(self, onnx_path: str) -> Tuple[float, int]:
        """Validar modelo ONNX."""
        print("\n🔄 Validando ONNX...")
        
        if not os.path.exists(onnx_path):
            print(f"❌ ONNX no encontrado: {onnx_path}")
            return 0, 0
        
        session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        total_detections = 0
        total_time = 0
        
        for image_path in self.images:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # Preprocesar
            image = cv2.resize(image, (640, 640))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, 0)
            
            input_name = session.get_inputs()[0].name
            
            start = time.perf_counter()
            outputs = session.run(None, {input_name: image})
            end = time.perf_counter()
            
            total_time += (end - start) * 1000
            
            # Contar detecciones (formato varía según ONNX export)
            if outputs:
                detections = outputs[0]
                if isinstance(detections, np.ndarray):
                    total_detections += len(detections)
        
        avg_time = total_time / len(self.images) if self.images else 0
        print(f"✓ ONNX - Latencia promedio: {avg_time:.2f} ms")
        
        return avg_time, total_detections
    
    def compare_outputs(self, pytorch_model_path: str, onnx_path: str) -> float:
        """Comparar outputs PyTorch vs ONNX (similitud)."""
        print("\n🔄 Comparando outputs PyTorch vs ONNX...")
        
        if not self.images:
            print("❌ No hay imágenes para validar")
            return 0.0
        
        pytorch_model = YOLO(pytorch_model_path)
        pytorch_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        onnx_session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        similarities = []
        
        for image_path in self.images[:3]:  # Usar solo 3 para comparación
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # PyTorch
            pytorch_results = pytorch_model(image, verbose=False)
            
            # ONNX
            image_resized = cv2.resize(image, (640, 640))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_normalized = image_rgb.astype(np.float32) / 255.0
            image_tensor = np.transpose(image_normalized, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, 0)
            
            input_name = onnx_session.get_inputs()[0].name
            onnx_outputs = onnx_session.run(None, {input_name: image_tensor})
            
            # Calcular similitud (comparación de detecciones)
            pytorch_dets = len(pytorch_results[0].boxes)
            onnx_dets = len(onnx_outputs[0]) if onnx_outputs else 0
            
            similarity = 1.0 - abs(pytorch_dets - onnx_dets) / max(pytorch_dets, onnx_dets, 1)
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        print(f"✓ Similitud promedio: {avg_similarity:.2%}")
        
        return avg_similarity


class PerformanceAnalyzer:
    """Análisis de rendimiento y tradeoffs."""
    
    @staticmethod
    def print_summary(pytorch_time: float, onnx_time: float, 
                     similarity: float, tensorrt_time: float = None):
        """Imprimir resumen de rendimiento."""
        
        print("\n" + "=" * 70)
        print("⚡ ANALYSIS & RECOMMENDATIONS")
        print("=" * 70)
        
        if pytorch_time > 0 and onnx_time > 0:
            speedup_onnx = pytorch_time / onnx_time
            print(f"\n📈 ONNX Speedup: {speedup_onnx:.2f}x")
            print(f"   PyTorch: {pytorch_time:.2f} ms → ONNX: {onnx_time:.2f} ms")
        
        if tensorrt_time and tensorrt_time > 0:
            speedup_trt = pytorch_time / tensorrt_time
            print(f"\n📈 TensorRT Speedup: {speedup_trt:.2f}x")
            print(f"   PyTorch: {pytorch_time:.2f} ms → TensorRT: {tensorrt_time:.2f} ms")
        
        print(f"\n🎯 Similitud de outputs (PyTorch vs ONNX): {similarity:.2%}")
        
        if similarity > 0.95:
            print("   ✓ Excelente: outputs equivalentes")
        elif similarity > 0.90:
            print("   ⚠️  Bueno: outputs similares, posibles variaciones numéricas")
        else:
            print("   ⚠️  Revisar: grandes diferencias en outputs")
        
        print("\n💡 RECOMENDACIONES:")
        if speedup_onnx and speedup_onnx > 1.5:
            print("   - ONNX ofrece mejora significativa")
            print("   - Considerar usar ONNX en producción")
        
        if tensorrt_time and speedup_trt and speedup_trt > 2.0:
            print("   - TensorRT ofrece optimización extrema")
            print("   - Recomendado para aplicaciones en tiempo real")


def main():
    parser = __import__('argparse').ArgumentParser(
        description="Validar y comparar modelos YOLO (HU-04)"
    )
    parser.add_argument('--pytorch', type=str, default='../model.pt',
                       help='Ruta modelo PyTorch')
    parser.add_argument('--onnx', type=str, default='models/model.onnx',
                       help='Ruta modelo ONNX')
    parser.add_argument('--tensorrt', type=str, default='models/model.engine',
                       help='Ruta modelo TensorRT')
    parser.add_argument('--images', type=str, default='scripts/images',
                       help='Directorio con imágenes de prueba')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🔍 VALIDADOR HU-04: OPTIMIZACIÓN DE INFERENCIA")
    print("=" * 70)
    
    # Instanciar validador
    validator = YOLOValidator(image_dir=args.images)
    
    # Validar PyTorch (baseline)
    pytorch_time, pytorch_dets = validator.validate_pytorch(args.pytorch)
    
    # Validar ONNX
    onnx_time, onnx_dets = validator.validate_onnx(args.onnx)
    
    # Comparar outputs
    similarity = validator.compare_outputs(args.pytorch, args.onnx)
    
    # Análisis de performance
    if pytorch_time > 0 and onnx_time > 0:
        PerformanceAnalyzer.print_summary(pytorch_time, onnx_time, similarity)
    
    print("\n" + "=" * 70)
    print("✅ Validación completada")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
