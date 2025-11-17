#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono> // Para medir el tiempo

// Cabeceras de OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> // Para blobFromImage

// Cabeceras de CUDA y TensorRT
#include <cuda_runtime_api.h>
#include <NvInfer.h>

using namespace nvinfer1;

// Logger de TensorRT (Boilerplate obligatorio)
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }
} gLogger;

// Función de Post-procesamiento
int postprocesar_cpp(float* output_buffer, float conf_threshold = 0.5, float nms_threshold = 0.45) {
    const int num_proposals = 6300;

    float* x_ptr = output_buffer;
    float* y_ptr = output_buffer + num_proposals;
    float* w_ptr = output_buffer + 2 * num_proposals;
    float* h_ptr = output_buffer + 3 * num_proposals;
    float* conf_ptr = output_buffer + 4 * num_proposals;

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;

    for (int p = 0; p < num_proposals; ++p) {
        if (conf_ptr[p] > conf_threshold) {
            int x1 = static_cast<int>(x_ptr[p] - w_ptr[p] / 2.0f);
            int y1 = static_cast<int>(y_ptr[p] - h_ptr[p] / 2.0f);
            int w = static_cast<int>(w_ptr[p]);
            int h = static_cast<int>(h_ptr[p]);
            boxes.emplace_back(x1, y1, w, h);
            confs.push_back(conf_ptr[p]);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, conf_threshold, nms_threshold, indices);

    return indices.size();
}

int main() {
    // Configuración
    const std::string video_path = "videos/prueba1.mp4";
    const std::string engine_path = "models/best_b1.engine";
    const int BATCH_SIZE = 1;

    // 1. Cargar el Motor TensorRT
    std::ifstream file(engine_path, std::ios::binary);

    if (!file.is_open()) { // .is_open() es correcto para std::ifstream
        std::cerr << "ERROR: No se pudo abrir el archivo del motor en: " << engine_path << std::endl;
        std::cerr << "Asegúrate de que el archivo existe y estás ejecutando esto desde la carpeta 'proyecto'." << std::endl;
        return -1;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    if (size == 0) {
        std::cerr << "ERROR: El archivo del motor está vacío (0 bytes): " << engine_path << std::endl;
        std::cerr << "Esto significa que el comando 'trtexec' de antes falló. Intenta borrarlo y generarlo de nuevo." << std::endl;
        file.close();
        return -1;
    }

    char* trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    IExecutionContext* context = engine->createExecutionContext();
    delete[] trtModelStream;
    
    if (!engine) {
        std::cerr << "ERROR: deserializeCudaEngine falló (posiblemente un motor corrupto)." << std::endl;
        return -1;
    }

    // 2. Asignar Buffers
    const int INPUT_SIZE = BATCH_SIZE * 3 * 480 * 640;
    const int OUTPUT_SIZE = BATCH_SIZE * 5 * 6300;

    void* buffers[2];
    cudaMalloc(&buffers[0], INPUT_SIZE * sizeof(float));
    cudaMalloc(&buffers[1], OUTPUT_SIZE * sizeof(float));
    
    std::vector<float> host_output_buffer(OUTPUT_SIZE);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    const char* input_name = engine->getIOTensorName(0);
    const char* output_name = engine->getIOTensorName(1);
    std::cout << "Motor de inferencia IO:" << std::endl;
    std::cout << "  Input name: " << input_name << std::endl;
    std::cout << "  Output name: " << output_name << std::endl;
    
    context->setTensorAddress(input_name, buffers[0]);
    context->setTensorAddress(output_name, buffers[1]);

    // 3. Abrir Video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error al abrir el video en: " << video_path << std::endl;
        return -1;
    }

    // 4. El Pipeline Secuencial
    int total_frames = 0;
    int total_faces = 0;
    std::vector<double> tiempos;

    std::cout << "Iniciando pipeline C++ (Secuencial)..." << std::endl;

    while (true) {
        cv::Mat frame;
        cap.read(frame);
        if (frame.empty()) {
            break;
        }

        auto start = std::chrono::steady_clock::now();

        // 1. Preprocesar
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(640, 480), cv::Scalar(), true, false);

        // 2. Inferir (TensorRT)
        cudaMemcpyAsync(buffers[0], blob.ptr<float>(), blob.total() * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueueV3(stream);
        cudaMemcpyAsync(host_output_buffer.data(), buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // 3. Postprocesar
        int faces_en_frame = postprocesar_cpp(host_output_buffer.data());

        auto end = std::chrono::steady_clock::now();
        tiempos.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);

        total_frames++;
        total_faces += faces_en_frame;
        std::cout << "Procesados " << total_frames << " frames...\r" << std::flush;
    }

    // 5. Limpieza y Reporte
    cap.release();
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    
    delete context;
    delete engine;
    delete runtime;

    double duracion_total_ms = 0;
    for (double t : tiempos) duracion_total_ms += t;
    double duracion_total_s = duracion_total_ms / 1000.0;
    
    double fps = total_frames / duracion_total_s;
    double latencia_ms = duracion_total_ms / total_frames;

    std::cout << "\n\n=== RESULTADOS (C++ SECUENCIAL: " << BATCH_SIZE << ") ===" << std::endl;
    std::cout << "Frames procesados: " << total_frames << std::endl;
    std::cout << "Rostros detectados: " << total_faces << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "FPS promedio (Throughput): " << fps << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Latencia promedio por FRAME: " << latencia_ms << " ms" << std::endl;

    return 0;
}