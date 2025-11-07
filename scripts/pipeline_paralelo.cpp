#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono> // Para medir el tiempo
#include <thread> // ¡NUEVO! Para paralelismo
#include <queue>  // ¡NUEVO! Para la cola
#include <mutex>  // ¡NUEVO! Para la cola
#include <condition_variable> // ¡NUEVO! Para la cola

// Cabeceras de OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> // Para blobFromImage

// Cabeceras de CUDA y TensorRT
#include <cuda_runtime_api.h>
#include <NvInfer.h>

using namespace nvinfer1;

// --- Logger de TensorRT (Sin cambios) ---
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }
} gLogger;

// --- ¡NUEVO! Una cola segura para hilos ---
template <typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(std::move(value));
        m_cond.notify_one();
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [this] { return !m_queue.empty(); });
        if (m_queue.empty()) { // Debería ser redundante, pero es seguro
            return false;
        }
        value = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

private:
    std::queue<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cond;
};

// --- Función de Post-procesamiento (Sin cambios) ---
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

// --- ¡NUEVO! Hilo Productor (CPU) ---
// Este hilo lee el video y preprocesa los frames
void productor_worker(const std::string& video_path, ThreadSafeQueue<cv::Mat>& prepro_queue) {
    std::cout << "[Productor] Hilo iniciado. Abriendo video..." << std::endl;
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[Productor] ERROR: No se pudo abrir el video." << std::endl;
        prepro_queue.push(cv::Mat()); // Envía un frame vacío para señalar el error
        return;
    }

    while (true) {
        cv::Mat frame;
        cap.read(frame);
        if (frame.empty()) {
            break; // Fin del video
        }

        // 1. Preprocesar (OpenCV C++)
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(640, 480), cv::Scalar(), true, false);
        
        // Poner el 'blob' listo en la cola
        prepro_queue.push(blob);
    }

    // Poner un 'blob' vacío en la cola para señalar el fin
    prepro_queue.push(cv::Mat());
    std::cout << "[Productor] Video terminado. Hilo finalizado." << std::endl;
    cap.release();
}

// --- Hilo Consumidor (Principal) ---
int main() {
    // --- Configuración ---
    const std::string video_path = "videos/prueba1.mp4";
    const std::string engine_path = "models/best_b1.engine";
    const int BATCH_SIZE = 1;

    // --- 1. Cargar el Motor TensorRT ---
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()) { /* ... (manejo de errores) ... */ return -1; }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    if (size == 0) { /* ... (manejo de errores) ... */ return -1; }
    char* trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    IExecutionContext* context = engine->createExecutionContext();
    delete[] trtModelStream;
    if (!engine) { /* ... (manejo de errores) ... */ return -1; }

    // --- 2. Asignar Buffers (Memoria) ---
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
    context->setTensorAddress(input_name, buffers[0]);
    context->setTensorAddress(output_name, buffers[1]);
    
    // --- 3. ¡NUEVO! Iniciar el Pipeline Paralelo ---
    ThreadSafeQueue<cv::Mat> prepro_queue;
    std::thread productor_hilo(productor_worker, video_path, std::ref(prepro_queue));

    // --- 4. El Pipeline (Consumidor) ---
    int total_frames = 0;
    int total_faces = 0;
    std::vector<double> tiempos;

    std::cout << "[Consumidor] Iniciando pipeline C++ (Paralelo)..." << std::endl;

    // ¡Empezar a medir el tiempo total AHORA!
    auto pipeline_start = std::chrono::steady_clock::now();

    while (true) {
        // Tomar un 'blob' preprocesado de la cola
        cv::Mat blob;
        prepro_queue.pop(blob); // Esto espera si la cola está vacía

        if (blob.empty()) {
            break; // El productor envió la señal de fin
        }

        // ¡Medir SOLO el trabajo de la GPU y el post-proceso!
        // El pre-proceso ya se hizo en el otro hilo.
        auto frame_start = std::chrono::steady_clock::now();

        // 2. Inferir (TensorRT)
        cudaMemcpyAsync(buffers[0], blob.ptr<float>(), blob.total() * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueueV3(stream);
        cudaMemcpyAsync(host_output_buffer.data(), buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // 3. Postprocesar (C++)
        int faces_en_frame = postprocesar_cpp(host_output_buffer.data());

        auto frame_end = std::chrono::steady_clock::now();
        tiempos.push_back(std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start).count() / 1000.0);

        total_frames++;
        total_faces += faces_en_frame;
        std::cout << "Procesados " << total_frames << " frames...\r" << std::flush;
    }

    auto pipeline_end = std::chrono::steady_clock::now();

    // --- 5. Limpieza y Reporte ---
    productor_hilo.join(); // Esperar a que el hilo productor termine
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete context;
    delete engine;
    delete runtime;

    // --- Reporte de Métricas ---
    double duracion_total_ms = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end - pipeline_start).count() / 1000.0;
    double duracion_total_s = duracion_total_ms / 1000.0;
    
    double fps = total_frames / duracion_total_s;
    
    // Calcular la latencia promedio del "consumidor"
    double latencia_consumidor_ms = 0;
    for (double t : tiempos) latencia_consumidor_ms += t;
    double latencia_avg_ms = latencia_consumidor_ms / tiempos.size();

    std::cout << "\n\n=== RESULTADOS (C++ PARALELO (hilos): " << BATCH_SIZE << ") ===" << std::endl;
    std::cout << "Frames procesados: " << total_frames << std::endl;
    std::cout << "Rostros detectados: " << total_faces << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "🔥 FPS promedio (Throughput Total): " << fps << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Latencia promedio (Infer + Post): " << latencia_avg_ms << " ms" << std::endl;

    return 0;
}