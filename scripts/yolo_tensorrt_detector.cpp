/*
 * YOLO Inference Pipeline with TensorRT + CUDA
 * HU-04: Optimización de Inferencia
 * 
 * Compilar:
 *   mkdir build && cd build
 *   cmake ..
 *   make -j$(nproc)
 * 
 * Ejecutar:
 *   ./yolo_tensorrt_detector <engine.trt> <video_path> [confidence_threshold]
 */

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Configuración
const int INPUT_W = 640;
const int INPUT_H = 640;
const int NUM_CLASSES = 1;  // Face detection = 1 clase
const float CONF_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.45f;

// Estructuras para detecciones
struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

// Logger para TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR) {
            cerr << "❌ [TensorRT ERROR]: " << msg << endl;
        } else if (severity == Severity::kWARNING) {
            cout << "⚠️  [TensorRT WARNING]: " << msg << endl;
        }
    }
} g_logger;

// Clase principal para detección con TensorRT
class YOLOTensorRTDetector {
private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    
    vector<void*> buffers;
    size_t input_size;
    size_t output_size;
    
    float* input_device;
    float* output_device;

public:
    YOLOTensorRTDetector(const string& engine_path) : runtime(nullptr), engine(nullptr), context(nullptr) {
        if (!loadEngine(engine_path)) {
            throw runtime_error("❌ No se pudo cargar el engine TensorRT");
        }
    }

    ~YOLOTensorRTDetector() {
        cleanup();
    }

private:
    bool loadEngine(const string& engine_path) {
        cout << "📂 Cargando engine TensorRT: " << engine_path << endl;

        // Leer archivo binario
        ifstream file(engine_path, ios::binary);
        if (!file.good()) {
            cerr << "❌ No se pudo abrir: " << engine_path << endl;
            return false;
        }

        file.seekg(0, ios::end);
        size_t size = file.tellg();
        file.seekg(0, ios::beg);

        vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        // Crear runtime y engine
        runtime = nvinfer1::createInferRuntime(g_logger);
        if (!runtime) {
            cerr << "❌ No se pudo crear runtime" << endl;
            return false;
        }

        engine = runtime->deserializeCudaEngine(engine_data.data(), size, nullptr);
        if (!engine) {
            cerr << "❌ No se pudo deserializar engine" << endl;
            return false;
        }

        context = engine->createExecutionContext();
        if (!context) {
            cerr << "❌ No se pudo crear contexto" << endl;
            return false;
        }

        // Obtener tamaños de entrada/salida
        int input_idx = engine->getBindingIndex("images");
        int output_idx = engine->getBindingIndex("output0");

        if (input_idx == -1 || output_idx == -1) {
            cerr << "❌ No se encontraron bindings esperados" << endl;
            return false;
        }

        auto input_dims = engine->getBindingDimensions(input_idx);
        auto output_dims = engine->getBindingDimensions(output_idx);

        input_size = 1;
        for (int i = 0; i < input_dims.nbDims; i++) {
            input_size *= input_dims.d[i];
        }

        output_size = 1;
        for (int i = 0; i < output_dims.nbDims; i++) {
            output_size *= output_dims.d[i];
        }

        cout << "✅ Engine cargado" << endl;
        cout << "   Input: " << input_size << " elementos" << endl;
        cout << "   Output: " << output_size << " elementos" << endl;

        // Asignar memoria GPU
        cudaMalloc(&input_device, input_size * sizeof(float));
        cudaMalloc(&output_device, output_size * sizeof(float));

        buffers.push_back(input_device);
        buffers.push_back(output_device);

        return true;
    }

    void cleanup() {
        if (context) context->destroy();
        if (engine) engine->destroy();
        if (runtime) runtime->destroy();

        cudaFree(input_device);
        cudaFree(output_device);
    }

public:
    vector<Detection> detect(const Mat& frame) {
        // Preprocesar imagen
        Mat blob = preprocessImage(frame);

        // Copiar a GPU
        cudaMemcpy(input_device, blob.data, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Ejecutar inferencia
        auto start_infer = chrono::steady_clock::now();

        void* bindings[] = { input_device, output_device };
        bool success = context->executeV2(bindings);

        auto end_infer = chrono::steady_clock::now();
        auto infer_time = chrono::duration_cast<chrono::milliseconds>(end_infer - start_infer).count();

        if (!success) {
            cerr << "❌ Inferencia fallida" << endl;
            return {};
        }

        cout << "⏱️  Tiempo inferencia: " << infer_time << " ms" << endl;

        // Copiar resultados a CPU
        vector<float> output_host(output_size);
        cudaMemcpy(output_host.data(), output_device, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        // Postprocesar detecciones
        vector<Detection> detections = postprocessDetections(output_host, frame.rows, frame.cols);

        return detections;
    }

private:
    Mat preprocessImage(const Mat& frame) {
        Mat resized;
        resize(frame, resized, Size(INPUT_W, INPUT_H));

        Mat blob;
        resized.convertTo(blob, CV_32F, 1.0 / 255.0);

        Mat flattened = blob.reshape(1, 1);
        return flattened;
    }

    vector<Detection> postprocessDetections(const vector<float>& output, int frame_h, int frame_w) {
        vector<Detection> detections;

        // Estructura de output YOLO: [x, y, w, h, obj_conf, class1, class2, ...]
        // Para YOLOv8: (num_detections, 6) -> [x, y, w, h, conf, class_id]

        size_t stride = 6;  // x, y, w, h, conf, class_id
        size_t num_detections = output.size() / stride;

        for (size_t i = 0; i < num_detections; i++) {
            size_t idx = i * stride;

            float conf = output[idx + 4];
            if (conf < CONF_THRESHOLD) continue;

            float x = output[idx + 0];
            float y = output[idx + 1];
            float w = output[idx + 2];
            float h = output[idx + 3];

            // Convertir de coordenadas normalizadas a píxeles
            float x1 = (x - w / 2) * frame_w;
            float y1 = (y - h / 2) * frame_h;
            float x2 = (x + w / 2) * frame_w;
            float y2 = (y + h / 2) * frame_h;

            detections.push_back({
                max(0.0f, x1),
                max(0.0f, y1),
                min((float)frame_w, x2),
                min((float)frame_h, y2),
                conf,
                (int)output[idx + 5]
            });
        }

        // Aplicar NMS (Non-Maximum Suppression)
        detections = applyNMS(detections, NMS_THRESHOLD);

        return detections;
    }

    vector<Detection> applyNMS(vector<Detection>& detections, float nms_threshold) {
        if (detections.empty()) return {};

        sort(detections.begin(), detections.end(), 
             [](const Detection& a, const Detection& b) {
                 return a.confidence > b.confidence;
             });

        vector<Detection> result;
        for (const auto& det : detections) {
            bool keep = true;
            for (const auto& res : result) {
                float iou = calculateIoU(det, res);
                if (iou > nms_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) result.push_back(det);
        }

        return result;
    }

    float calculateIoU(const Detection& a, const Detection& b) {
        float inter_x1 = max(a.x1, b.x1);
        float inter_y1 = max(a.y1, b.y1);
        float inter_x2 = min(a.x2, b.x2);
        float inter_y2 = min(a.y2, b.y2);

        float inter = max(0.0f, inter_x2 - inter_x1) * max(0.0f, inter_y2 - inter_y1);
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        float union_area = area_a + area_b - inter;

        return inter / (union_area + 1e-6);
    }
};

// Función para visualizar detecciones
void drawDetections(Mat& frame, const vector<Detection>& detections) {
    for (const auto& det : detections) {
        Rect rect(Point(det.x1, det.y1), Point(det.x2, det.y2));
        rectangle(frame, rect, Scalar(0, 255, 0), 2);

        string label = format("Face: %.2f", det.confidence);
        putText(frame, label, Point(det.x1, det.y1 - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }
}

// Main
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " <engine.trt> <video_path> [confidence]" << endl;
        return 1;
    }

    string engine_path = argv[1];
    string video_path = argv[2];
    float confidence = argc > 3 ? stof(argv[3]) : 0.5f;

    cout << "\n" << string(70, '=') << endl;
    cout << "🚀 YOLO TensorRT CUDA Detector (HU-04)" << endl;
    cout << string(70, '=') << "\n";

    try {
        // Cargar modelo TensorRT
        YOLOTensorRTDetector detector(engine_path);

        // Abrir video
        VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            cerr << "❌ No se pudo abrir: " << video_path << endl;
            return 1;
        }

        Mat frame;
        int frame_count = 0;
        int total_faces = 0;
        double total_time = 0;

        while (cap.read(frame)) {
            frame_count++;

            auto start = chrono::steady_clock::now();

            // Detectar rostros
            vector<Detection> detections = detector.detect(frame);

            auto end = chrono::steady_clock::now();
            total_time += chrono::duration_cast<chrono::milliseconds>(end - start).count();

            total_faces += detections.size();

            // Visualizar
            drawDetections(frame, detections);

            putText(frame, format("Frame: %d | Faces: %lu", frame_count, detections.size()),
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 2);

            imshow("YOLO TensorRT Detector", frame);

            if (waitKey(1) == 27) break;  // ESC para salir

            if (frame_count % 30 == 0) {
                cout << "   Frames procesados: " << frame_count << endl;
            }
        }

        cap.release();
        destroyAllWindows();

        // Resultados
        double fps = frame_count / (total_time / 1000.0);

        cout << "\n" << string(70, '=') << endl;
        cout << "📊 RESULTADOS" << endl;
        cout << string(70, '=') << endl;
        cout << "Frames procesados: " << frame_count << endl;
        cout << "Rostros detectados: " << total_faces << endl;
        cout << "FPS promedio: " << fixed << setprecision(2) << fps << endl;
        cout << "Latencia promedio: " << (total_time / frame_count) << " ms/frame" << endl;
        cout << string(70, '=') << "\n";

    } catch (const exception& e) {
        cerr << "❌ Excepción: " << e.what() << endl;
        return 1;
    }

    return 0;
}
