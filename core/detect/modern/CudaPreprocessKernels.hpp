#pragma once

#ifdef USE_GPU_ACCELERATION
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <memory>

namespace pv {

/**
 * CUDA kernels for high-performance image preprocessing
 * Optimized for YOLO inference pipeline with maximum throughput
 */
class CudaPreprocessKernels {
public:
    struct PreprocessConfig {
        int inputWidth = 1920;
        int inputHeight = 1080;
        int outputWidth = 640;
        int outputHeight = 640;
        float meanR = 0.0f;
        float meanG = 0.0f;
        float meanB = 0.0f;
        float stdR = 1.0f;
        float stdG = 1.0f;
        float stdB = 1.0f;
        bool maintainAspectRatio = true;
        bool useLetterbox = true;
    };

    CudaPreprocessKernels();
    ~CudaPreprocessKernels();

    // Initialize kernels with configuration
    bool initialize(const PreprocessConfig& config);
    
    // Main preprocessing pipeline (all operations in single kernel call)
    bool preprocessImage(
        const cv::cuda::GpuMat& input,
        float* output,
        cudaStream_t stream = nullptr
    );
    
    // Individual operations (for debugging/profiling)
    bool resizeWithLetterbox(
        const cv::cuda::GpuMat& input,
        cv::cuda::GpuMat& output,
        cudaStream_t stream = nullptr
    );
    
    bool normalizeAndConvert(
        const cv::cuda::GpuMat& input,
        float* output,
        cudaStream_t stream = nullptr
    );
    
    // Performance monitoring
    double getLastPreprocessTime() const { return lastPreprocessTime; }
    uint64_t getTotalOperations() const { return totalOperations; }

private:
    void allocateGpuMemory();
    void deallocateGpuMemory();

    PreprocessConfig config;
    bool initialized;
    
    // GPU memory buffers
    float* d_resized;
    float* d_normalized;
    uint8_t* d_temp;
    
    // Performance tracking
    mutable double lastPreprocessTime;
    mutable uint64_t totalOperations;
    
    // CUDA events for timing
    cudaEvent_t start_event, stop_event;
};

// CUDA kernel declarations
extern "C" {
    void launch_resize_letterbox_kernel(
        const uint8_t* input,
        float* output,
        int input_width,
        int input_height,
        int output_width,
        int output_height,
        float scale_x,
        float scale_y,
        int offset_x,
        int offset_y,
        float mean_r,
        float mean_g,
        float mean_b,
        float std_r,
        float std_g,
        float std_b,
        cudaStream_t stream
    );
    
    void launch_bgr_to_rgb_kernel(
        const uint8_t* input,
        uint8_t* output,
        int width,
        int height,
        cudaStream_t stream
    );
    
    void launch_normalize_kernel(
        const uint8_t* input,
        float* output,
        int width,
        int height,
        int channels,
        float mean_r,
        float mean_g,
        float mean_b,
        float std_r,
        float std_g,
        float std_b,
        cudaStream_t stream
    );
    
    void launch_hwc_to_chw_kernel(
        const float* input,
        float* output,
        int width,
        int height,
        int channels,
        cudaStream_t stream
    );
}

} // namespace pv

#endif // USE_GPU_ACCELERATION