#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#ifdef USE_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#endif

#ifdef USE_GPU_ACCELERATION
#include <cuda_runtime.h>
#endif

#include "../../util/Types.hpp"

#ifdef USE_GPU_ACCELERATION
#include "CudaPreprocessKernels.hpp"
#endif

namespace pv {

/**
 * High-performance TensorRT inference engine for ball detection
 * Optimized for maximum throughput with GPU-only processing pipeline
 */
class TensorRtBallDetector {
public:
    struct Config {
        std::string modelPath = "models/yolo_ball_detection.onnx";
        std::string engineCachePath = "models/yolo_ball_detection.trt";
        int maxBatchSize = 1;
        int inputWidth = 640;
        int inputHeight = 640;
        float confThreshold = 0.5f;
        float nmsThreshold = 0.4f;
        bool useFP16 = true;
        bool useDLACore = false;
        int workspaceSize = 256 * 1024 * 1024; // 256MB
    };

    struct Detection {
        float x, y, w, h;    // Bounding box
        float confidence;    // Detection confidence
        int classId;         // Class ID (0 = ball)
        
        // Convert to Ball structure
        Ball toBall() const {
            Ball b;
            b.c = cv::Point2f(x + w/2, y + h/2);
            b.r = std::max(w, h) / 2.0f;
            b.label = classId;
            b.stripeScore = confidence;
            return b;
        }
    };

    TensorRtBallDetector();
    ~TensorRtBallDetector();

    // Initialization and model loading
    bool initialize(const Config& config);
    bool loadModel(const std::string& modelPath);
    bool buildEngine();
    bool loadEngine();
    bool saveEngine();

    // Main inference pipeline
    std::vector<Detection> detect(const cv::cuda::GpuMat& image);
    std::vector<Ball> detectBalls(const cv::cuda::GpuMat& image);
    
    // Batch processing for higher throughput
    std::vector<std::vector<Detection>> detectBatch(const std::vector<cv::cuda::GpuMat>& images);

    // Performance monitoring
    double getLastInferenceTime() const { return lastInferenceTime; }
    double getAverageInferenceTime() const { return avgInferenceTime; }
    uint64_t getTotalInferences() const { return totalInferences; }
    bool isReady() const { return engineReady; }

private:
    Config config;
    bool engineReady;

#ifdef USE_TENSORRT
    // TensorRT components
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // Network information
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<nvinfer1::Dims> inputDims;
    std::vector<nvinfer1::Dims> outputDims;
    
    // GPU memory management
    std::vector<void*> bindings;
    std::vector<size_t> bindingSizes;
    void* d_input;
    void* d_output;
    cudaStream_t stream;
    
    // TensorRT logger
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };
    std::unique_ptr<Logger> logger;
    
    // Helper methods
    bool allocateBuffers();
    void deallocateBuffers();
    bool setupBindings();
    
    // Engine building
    std::unique_ptr<nvinfer1::ICudaEngine> buildEngineFromOnnx();
    std::unique_ptr<nvinfer1::ICudaEngine> loadEngineFromFile();
    
    // Post-processing
    std::vector<Detection> postProcess(float* output, int numDetections);
    std::vector<Detection> applyNMS(const std::vector<Detection>& detections);
#endif

    // Preprocessing
#ifdef USE_GPU_ACCELERATION
    std::unique_ptr<CudaPreprocessKernels> preprocessor;
#endif
    
    // Performance tracking
    mutable double lastInferenceTime;
    mutable double avgInferenceTime;
    mutable uint64_t totalInferences;
    
    // Timing
    cudaEvent_t startEvent, stopEvent;
};

} // namespace pv