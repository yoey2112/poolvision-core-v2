#include "CudaPreprocessKernels.hpp"

#ifdef USE_GPU_ACCELERATION

#include <iostream>
#include <chrono>

namespace pv {

CudaPreprocessKernels::CudaPreprocessKernels() 
    : initialized(false)
    , d_resized(nullptr)
    , d_normalized(nullptr)
    , d_temp(nullptr)
    , lastPreprocessTime(0.0)
    , totalOperations(0)
{
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
}

CudaPreprocessKernels::~CudaPreprocessKernels() {
    deallocateGpuMemory();
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

bool CudaPreprocessKernels::initialize(const PreprocessConfig& cfg) {
    config = cfg;
    
    try {
        allocateGpuMemory();
        initialized = true;
        std::cout << "CUDA preprocessing kernels initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize CUDA kernels: " << e.what() << std::endl;
        return false;
    }
}

bool CudaPreprocessKernels::preprocessImage(
    const cv::cuda::GpuMat& input,
    float* output,
    cudaStream_t stream) {
    
    if (!initialized) {
        std::cerr << "Kernels not initialized" << std::endl;
        return false;
    }
    
    cudaEventRecord(start_event, stream);
    
    try {
        // Calculate scaling and letterbox parameters
        float scale_x = static_cast<float>(config.outputWidth) / input.cols;
        float scale_y = static_cast<float>(config.outputHeight) / input.rows;
        
        int offset_x = 0, offset_y = 0;
        
        if (config.maintainAspectRatio) {
            float scale = std::min(scale_x, scale_y);
            scale_x = scale_y = scale;
            
            if (config.useLetterbox) {
                int scaled_width = static_cast<int>(input.cols * scale);
                int scaled_height = static_cast<int>(input.rows * scale);
                offset_x = (config.outputWidth - scaled_width) / 2;
                offset_y = (config.outputHeight - scaled_height) / 2;
            }
        }
        
        // Launch combined preprocessing kernel
        launch_resize_letterbox_kernel(
            input.ptr<uint8_t>(),
            output,
            input.cols,
            input.rows,
            config.outputWidth,
            config.outputHeight,
            scale_x,
            scale_y,
            offset_x,
            offset_y,
            config.meanR,
            config.meanG,
            config.meanB,
            config.stdR,
            config.stdG,
            config.stdB,
            stream
        );
        
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
        
        // Calculate execution time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        lastPreprocessTime = static_cast<double>(milliseconds);
        totalOperations++;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Preprocessing error: " << e.what() << std::endl;
        return false;
    }
}

bool CudaPreprocessKernels::resizeWithLetterbox(
    const cv::cuda::GpuMat& input,
    cv::cuda::GpuMat& output,
    cudaStream_t stream) {
    
    if (!initialized) return false;
    
    // Ensure output is properly sized
    output.create(config.outputHeight, config.outputWidth, CV_8UC3);
    
    // Calculate scale and offset
    float scale = std::min(
        static_cast<float>(config.outputWidth) / input.cols,
        static_cast<float>(config.outputHeight) / input.rows
    );
    
    int scaled_width = static_cast<int>(input.cols * scale);
    int scaled_height = static_cast<int>(input.rows * scale);
    int offset_x = (config.outputWidth - scaled_width) / 2;
    int offset_y = (config.outputHeight - scaled_height) / 2;
    
    // Use OpenCV's GPU resize as fallback for now
    cv::cuda::GpuMat resized;
    cv::cuda::resize(input, resized, cv::Size(scaled_width, scaled_height), 0, 0, cv::INTER_LINEAR, stream);
    
    // Create letterbox with padding
    output.setTo(cv::Scalar(114, 114, 114), stream); // Gray padding
    cv::Rect roi(offset_x, offset_y, scaled_width, scaled_height);
    resized.copyTo(output(roi), stream);
    
    return true;
}

bool CudaPreprocessKernels::normalizeAndConvert(
    const cv::cuda::GpuMat& input,
    float* output,
    cudaStream_t stream) {
    
    if (!initialized) return false;
    
    launch_normalize_kernel(
        input.ptr<uint8_t>(),
        output,
        input.cols,
        input.rows,
        input.channels(),
        config.meanR,
        config.meanG,
        config.meanB,
        config.stdR,
        config.stdG,
        config.stdB,
        stream
    );
    
    return true;
}

void CudaPreprocessKernels::allocateGpuMemory() {
    size_t output_size = config.outputWidth * config.outputHeight * 3 * sizeof(float);
    size_t temp_size = config.inputWidth * config.inputHeight * 3;
    
    cudaMalloc(&d_resized, output_size);
    cudaMalloc(&d_normalized, output_size);
    cudaMalloc(&d_temp, temp_size);
    
    // Check for allocation errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)));
    }
}

void CudaPreprocessKernels::deallocateGpuMemory() {
    if (d_resized) {
        cudaFree(d_resized);
        d_resized = nullptr;
    }
    if (d_normalized) {
        cudaFree(d_normalized);
        d_normalized = nullptr;
    }
    if (d_temp) {
        cudaFree(d_temp);
        d_temp = nullptr;
    }
}

} // namespace pv

#endif // USE_GPU_ACCELERATION