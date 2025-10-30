#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main() {
    string video_path = "videos/prueba1.mp4";
    string model_path = "best.onnx";

    // 1. Cargar modelo ONNX
    Net net = readNetFromONNX(model_path);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // 2. Capturar video
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "No se pudo abrir el video." << endl;
        return -1;
    }

    Mat frame;
    int total_frames = 0;
    int total_faces = 0;
    double total_time = 0.0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto start = chrono::steady_clock::now();

        // 3. Preprocesamiento
        Mat blob = blobFromImage(frame, 1.0 / 255.0, Size(640, 480), Scalar(), true, false);

        // 4. Inferencia
        net.setInput(blob);
        Mat detections = net.forward();

        // 5. Postprocesamiento básico
        int faces_detected = 0;
        for (int i = 0; i < detections.rows; i++) {
            float conf = detections.at<float>(i, 4);
            if (conf > 0.5) {
                int x1 = (int)(detections.at<float>(i, 0) * frame.cols);
                int y1 = (int)(detections.at<float>(i, 1) * frame.rows);
                int x2 = (int)(detections.at<float>(i, 2) * frame.cols);
                int y2 = (int)(detections.at<float>(i, 3) * frame.rows);

                rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);
                faces_detected++;
            }
        }

        total_faces += faces_detected;
        total_frames++;

        auto end = chrono::steady_clock::now();
        total_time += chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.0;

        imshow("Detección Secuencial", frame);
        if (waitKey(1) == 27) break; // Esc para salir
    }

    cap.release();
    destroyAllWindows();

    double fps = total_frames / total_time;
    double latency = total_time / total_frames;

    cout << "\n=== RESULTADOS SECUENCIALES ===" << endl;
    cout << "Frames procesados: " << total_frames << endl;
    cout << "Rostros detectados: " << total_faces << endl;
    cout << "FPS promedio: " << fps << endl;
    cout << "Latencia promedio por frame: " << latency << " s" << endl;

    return 0;
}
