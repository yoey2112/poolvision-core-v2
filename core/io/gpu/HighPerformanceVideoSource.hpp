#pragma once

#ifdef USE_GPU_ACCELERATION

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

#ifdef USE_NVDEC
#include <cuda_runtime.h>
#include <cuda.h>
// NVDEC headers would be included here
// #include "NvDecoder/NvDecoder.h"
#endif

namespace pv {

/**
 * High-performance video source with NVDEC hardware acceleration
 * Designed for maximum FPS processing with GPU memory optimization
 */
class HighPerformanceVideoSource {
public:
    struct PerformanceMetrics {
        std::atomic<uint64_t> framesDecoded{0};
        std::atomic<uint64_t> framesCaptured{0};
        std::atomic<uint64_t> framesDropped{0};
        std::atomic<double> avgDecodeTime{0.0};
        std::atomic<double> avgCaptureTime{0.0};
        
        double getDecodeFPS() const {
            return avgDecodeTime.load() > 0 ? 1000.0 / avgDecodeTime.load() : 0.0;
        }
        
        double getCaptureFPS() const {
            return avgCaptureTime.load() > 0 ? 1000.0 / avgCaptureTime.load() : 0.0;
        }
    };

    HighPerformanceVideoSource();
    ~HighPerformanceVideoSource();
    
    // Core functionality
    bool open(const std::string &source);
    bool read(cv::cuda::GpuMat &frame);  // GPU memory output
    bool readCPU(cv::Mat &frame);        // CPU memory output (fallback)
    void release();
    
    // Performance monitoring
    double fps() const;
    const PerformanceMetrics& getMetrics() const { return metrics; }
    bool isHardwareAccelerated() const { return useHardwareAcceleration; }
    
    // Configuration
    void setTargetResolution(const cv::Size &size) { targetResolution = size; }
    void setMaxBufferSize(int size) { maxBufferSize = size; }
    void setUseHardwareAcceleration(bool enable);

private:
    void captureLoop();
    void processFrame();
    bool initializeNVDEC();
    bool initializeFallback();
    void configureFallbackCapture();
    void cleanupResources();
    
    // Hardware acceleration state
    bool useHardwareAcceleration;
    bool hardwareReady;
    
    // OpenCV fallback
    cv::VideoCapture fallbackCapture;
    
    // NVDEC components
#ifdef USE_NVDEC
    // std::unique_ptr<NvDecoder> nvDecoder;
    CUcontext cuContext;
    CUdevice cuDevice;
    cudaStream_t cudaStream;
#endif

    // Threading and synchronization
    std::atomic<bool> running;
    std::atomic<bool> captureReady;
    std::thread captureThread;
    std::thread processThread;
    
    // Frame buffer management
    struct FrameBuffer {
        cv::cuda::GpuMat gpuFrame;
        cv::Mat cpuFrame;
        int64_t timestamp;
        bool valid;
    };
    
    static constexpr int DEFAULT_BUFFER_SIZE = 8;
    int maxBufferSize;
    std::queue<FrameBuffer> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCondition;
    
    // Configuration
    std::string sourceString;
    cv::Size targetResolution;
    double sourceFPS;
    
    // Performance tracking
    mutable PerformanceMetrics metrics;
    std::chrono::high_resolution_clock::time_point lastFrameTime;
};

/**
 * Factory function to create the best available video source
 */
std::unique_ptr<HighPerformanceVideoSource> createOptimalVideoSource(
    const std::string &source,
    bool preferHardwareAcceleration = true);

} // namespace pv

#endif // USE_GPU_ACCELERATION