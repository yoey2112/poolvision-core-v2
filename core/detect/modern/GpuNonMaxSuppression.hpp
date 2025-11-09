#pragma once

#ifdef USE_GPU_ACCELERATION

#include <vector>
#include <cuda_runtime.h>

namespace pv {

/**
 * GPU-accelerated Non-Maximum Suppression for real-time object detection
 * Optimized for high-throughput ball detection pipeline
 */
class GpuNonMaxSuppression {
public:
    struct Detection {
        float x, y, w, h;
        float confidence;
        int classId;
    };

    struct Config {
        float nmsThreshold = 0.4f;
        float confThreshold = 0.5f;
        int maxDetections = 1000;
        int maxOutputDetections = 100;
    };

    GpuNonMaxSuppression();
    ~GpuNonMaxSuppression();

    bool initialize(const Config& config);
    
    // GPU-only NMS (input and output on GPU)
    int performNMS(
        const float* d_detections,
        int numDetections,
        float* d_outputDetections,
        cudaStream_t stream = nullptr
    );
    
    // CPU-GPU hybrid NMS (for compatibility)
    std::vector<Detection> performNMS(
        const std::vector<Detection>& detections,
        cudaStream_t stream = nullptr
    );

    // Performance monitoring
    double getLastNMSTime() const { return lastNMSTime; }
    uint64_t getTotalOperations() const { return totalOperations; }

private:
    Config config;
    bool initialized;

    // GPU memory buffers
    float* d_sortedDetections;
    int* d_indices;
    bool* d_suppressed;
    float* d_iouMatrix;
    
    // Performance tracking
    mutable double lastNMSTime;
    mutable uint64_t totalOperations;
    
    // CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    
    // Helper methods
    void allocateGpuMemory();
    void deallocateGpuMemory();
};

// CUDA kernel declarations
extern "C" {
    void launch_compute_iou_kernel(
        const float* detections,
        float* iouMatrix,
        int numDetections,
        cudaStream_t stream
    );
    
    void launch_nms_kernel(
        const float* detections,
        const float* iouMatrix,
        bool* suppressed,
        int numDetections,
        float nmsThreshold,
        cudaStream_t stream
    );
    
    void launch_filter_detections_kernel(
        const float* inputDetections,
        const bool* suppressed,
        float* outputDetections,
        int* outputCount,
        int numDetections,
        float confThreshold,
        cudaStream_t stream
    );
    
    void launch_sort_detections_kernel(
        float* detections,
        int* indices,
        int numDetections,
        cudaStream_t stream
    );
}

} // namespace pv

#endif // USE_GPU_ACCELERATION