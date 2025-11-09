#include "GpuNonMaxSuppression.hpp"

#ifdef USE_GPU_ACCELERATION

#include <iostream>
#include <algorithm>
#include <chrono>

namespace pv {

GpuNonMaxSuppression::GpuNonMaxSuppression()
    : initialized(false)
    , d_sortedDetections(nullptr)
    , d_indices(nullptr)
    , d_suppressed(nullptr)
    , d_iouMatrix(nullptr)
    , lastNMSTime(0.0)
    , totalOperations(0)
{
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
}

GpuNonMaxSuppression::~GpuNonMaxSuppression() {
    deallocateGpuMemory();
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

bool GpuNonMaxSuppression::initialize(const Config& cfg) {
    config = cfg;
    
    try {
        allocateGpuMemory();
        initialized = true;
        std::cout << "GPU NMS initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize GPU NMS: " << e.what() << std::endl;
        return false;
    }
}

int GpuNonMaxSuppression::performNMS(
    const float* d_detections,
    int numDetections,
    float* d_outputDetections,
    cudaStream_t stream)
{
    if (!initialized || numDetections == 0) {
        return 0;
    }
    
    cudaEventRecord(startEvent, stream);
    
    try {
        // Sort detections by confidence (descending)
        launch_sort_detections_kernel(
            const_cast<float*>(d_detections),
            d_indices,
            numDetections,
            stream
        );
        
        // Compute IoU matrix
        launch_compute_iou_kernel(
            d_detections,
            d_iouMatrix,
            numDetections,
            stream
        );
        
        // Apply NMS
        launch_nms_kernel(
            d_detections,
            d_iouMatrix,
            d_suppressed,
            numDetections,
            config.nmsThreshold,
            stream
        );
        
        // Filter and pack results
        int outputCount = 0;
        int* d_outputCount;
        cudaMalloc(&d_outputCount, sizeof(int));
        cudaMemset(d_outputCount, 0, sizeof(int));
        
        launch_filter_detections_kernel(
            d_detections,
            d_suppressed,
            d_outputDetections,
            d_outputCount,
            numDetections,
            config.confThreshold,
            stream
        );
        
        // Get output count
        cudaMemcpyAsync(&outputCount, d_outputCount, sizeof(int), 
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        cudaFree(d_outputCount);
        
        cudaEventRecord(stopEvent, stream);
        cudaEventSynchronize(stopEvent);
        
        // Update performance metrics
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
        lastNMSTime = static_cast<double>(milliseconds);
        totalOperations++;
        
        return std::min(outputCount, config.maxOutputDetections);
        
    } catch (const std::exception& e) {
        std::cerr << "GPU NMS error: " << e.what() << std::endl;
        return 0;
    }
}

std::vector<GpuNonMaxSuppression::Detection> GpuNonMaxSuppression::performNMS(
    const std::vector<Detection>& detections,
    cudaStream_t stream)
{
    if (detections.empty()) {
        return {};
    }
    
    // Convert to GPU format
    std::vector<float> hostDetections(detections.size() * 6);
    for (size_t i = 0; i < detections.size(); i++) {
        int offset = i * 6;
        hostDetections[offset + 0] = detections[i].x;
        hostDetections[offset + 1] = detections[i].y;
        hostDetections[offset + 2] = detections[i].w;
        hostDetections[offset + 3] = detections[i].h;
        hostDetections[offset + 4] = detections[i].confidence;
        hostDetections[offset + 5] = static_cast<float>(detections[i].classId);
    }
    
    // Copy to GPU
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, hostDetections.size() * sizeof(float));
    cudaMalloc(&d_output, config.maxOutputDetections * 6 * sizeof(float));
    
    cudaMemcpyAsync(d_input, hostDetections.data(), 
                   hostDetections.size() * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
    
    // Perform GPU NMS
    int outputCount = performNMS(d_input, static_cast<int>(detections.size()), d_output, stream);
    
    // Copy results back
    std::vector<float> outputDetections(outputCount * 6);
    cudaMemcpyAsync(outputDetections.data(), d_output,
                   outputCount * 6 * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Convert back to Detection format
    std::vector<Detection> result;
    result.reserve(outputCount);
    
    for (int i = 0; i < outputCount; i++) {
        int offset = i * 6;
        Detection det;
        det.x = outputDetections[offset + 0];
        det.y = outputDetections[offset + 1];
        det.w = outputDetections[offset + 2];
        det.h = outputDetections[offset + 3];
        det.confidence = outputDetections[offset + 4];
        det.classId = static_cast<int>(outputDetections[offset + 5]);
        result.push_back(det);
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}

void GpuNonMaxSuppression::allocateGpuMemory() {
    size_t detectionSize = config.maxDetections * 6 * sizeof(float);
    size_t indexSize = config.maxDetections * sizeof(int);
    size_t suppressedSize = config.maxDetections * sizeof(bool);
    size_t iouMatrixSize = config.maxDetections * config.maxDetections * sizeof(float);
    
    cudaMalloc(&d_sortedDetections, detectionSize);
    cudaMalloc(&d_indices, indexSize);
    cudaMalloc(&d_suppressed, suppressedSize);
    cudaMalloc(&d_iouMatrix, iouMatrixSize);
    
    // Check for allocation errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("GPU NMS memory allocation failed: " + 
                               std::string(cudaGetErrorString(err)));
    }
}

void GpuNonMaxSuppression::deallocateGpuMemory() {
    if (d_sortedDetections) {
        cudaFree(d_sortedDetections);
        d_sortedDetections = nullptr;
    }
    if (d_indices) {
        cudaFree(d_indices);
        d_indices = nullptr;
    }
    if (d_suppressed) {
        cudaFree(d_suppressed);
        d_suppressed = nullptr;
    }
    if (d_iouMatrix) {
        cudaFree(d_iouMatrix);
        d_iouMatrix = nullptr;
    }
}

} // namespace pv

#endif // USE_GPU_ACCELERATION