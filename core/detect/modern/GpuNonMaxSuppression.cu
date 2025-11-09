#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// Compute IoU matrix for all detection pairs
__global__ void compute_iou_kernel(
    const float* __restrict__ detections,
    float* __restrict__ iouMatrix,
    int numDetections)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= numDetections || j >= numDetections) return;
    
    // Each detection: [x, y, w, h, conf, class]
    int offset_i = i * 6;
    int offset_j = j * 6;
    
    float x1_i = detections[offset_i + 0];
    float y1_i = detections[offset_i + 1];
    float w_i = detections[offset_i + 2];
    float h_i = detections[offset_i + 3];
    float x2_i = x1_i + w_i;
    float y2_i = y1_i + h_i;
    
    float x1_j = detections[offset_j + 0];
    float y1_j = detections[offset_j + 1];
    float w_j = detections[offset_j + 2];
    float h_j = detections[offset_j + 3];
    float x2_j = x1_j + w_j;
    float y2_j = y1_j + h_j;
    
    // Compute intersection
    float xx1 = fmaxf(x1_i, x1_j);
    float yy1 = fmaxf(y1_i, y1_j);
    float xx2 = fminf(x2_i, x2_j);
    float yy2 = fminf(y2_i, y2_j);
    
    float intersection = 0.0f;
    if (xx2 > xx1 && yy2 > yy1) {
        intersection = (xx2 - xx1) * (yy2 - yy1);
    }
    
    // Compute union
    float area_i = w_i * h_i;
    float area_j = w_j * h_j;
    float union_area = area_i + area_j - intersection;
    
    // Compute IoU
    float iou = (union_area > 0.0f) ? (intersection / union_area) : 0.0f;
    
    iouMatrix[i * numDetections + j] = iou;
}

// Apply Non-Maximum Suppression
__global__ void nms_kernel(
    const float* __restrict__ detections,
    const float* __restrict__ iouMatrix,
    bool* __restrict__ suppressed,
    int numDetections,
    float nmsThreshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numDetections) return;
    
    // Initialize suppressed status
    if (idx == 0) {
        suppressed[0] = false; // Best detection is never suppressed
    } else {
        suppressed[idx] = false;
        
        // Check against all previous detections (sorted by confidence)
        for (int i = 0; i < idx; i++) {
            if (!suppressed[i]) {
                float iou = iouMatrix[i * numDetections + idx];
                if (iou > nmsThreshold) {
                    suppressed[idx] = true;
                    break;
                }
            }
        }
    }
}

// Filter and pack non-suppressed detections
__global__ void filter_detections_kernel(
    const float* __restrict__ inputDetections,
    const bool* __restrict__ suppressed,
    float* __restrict__ outputDetections,
    int* __restrict__ outputCount,
    int numDetections,
    float confThreshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numDetections) return;
    
    int input_offset = idx * 6;
    float confidence = inputDetections[input_offset + 4];
    
    // Check if detection passes confidence threshold and is not suppressed
    if (confidence >= confThreshold && !suppressed[idx]) {
        // Atomic increment to get output position
        int output_idx = atomicAdd(outputCount, 1);
        int output_offset = output_idx * 6;
        
        // Copy detection data
        outputDetections[output_offset + 0] = inputDetections[input_offset + 0];
        outputDetections[output_offset + 1] = inputDetections[input_offset + 1];
        outputDetections[output_offset + 2] = inputDetections[input_offset + 2];
        outputDetections[output_offset + 3] = inputDetections[input_offset + 3];
        outputDetections[output_offset + 4] = inputDetections[input_offset + 4];
        outputDetections[output_offset + 5] = inputDetections[input_offset + 5];
    }
}

// Comparison function for sorting detections by confidence
struct ConfidenceComparator {
    const float* detections;
    
    __host__ __device__ ConfidenceComparator(const float* det) : detections(det) {}
    
    __host__ __device__ bool operator()(int a, int b) const {
        float conf_a = detections[a * 6 + 4];
        float conf_b = detections[b * 6 + 4];
        return conf_a > conf_b; // Descending order
    }
};

// Sort detections by confidence (using Thrust)
__global__ void prepare_sort_indices_kernel(int* indices, int numDetections) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numDetections) {
        indices[idx] = idx;
    }
}

// Reorder detections based on sorted indices
__global__ void reorder_detections_kernel(
    const float* __restrict__ inputDetections,
    const int* __restrict__ sortedIndices,
    float* __restrict__ outputDetections,
    int numDetections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numDetections) return;
    
    int source_idx = sortedIndices[idx];
    int source_offset = source_idx * 6;
    int dest_offset = idx * 6;
    
    // Copy detection data
    for (int i = 0; i < 6; i++) {
        outputDetections[dest_offset + i] = inputDetections[source_offset + i];
    }
}

// Host wrapper functions
extern "C" {

void launch_compute_iou_kernel(
    const float* detections,
    float* iouMatrix,
    int numDetections,
    cudaStream_t stream)
{
    dim3 block_size(16, 16);
    dim3 grid_size(
        (numDetections + block_size.x - 1) / block_size.x,
        (numDetections + block_size.y - 1) / block_size.y
    );
    
    compute_iou_kernel<<<grid_size, block_size, 0, stream>>>(
        detections, iouMatrix, numDetections
    );
}

void launch_nms_kernel(
    const float* detections,
    const float* iouMatrix,
    bool* suppressed,
    int numDetections,
    float nmsThreshold,
    cudaStream_t stream)
{
    dim3 block_size(256);
    dim3 grid_size((numDetections + block_size.x - 1) / block_size.x);
    
    nms_kernel<<<grid_size, block_size, 0, stream>>>(
        detections, iouMatrix, suppressed, numDetections, nmsThreshold
    );
}

void launch_filter_detections_kernel(
    const float* inputDetections,
    const bool* suppressed,
    float* outputDetections,
    int* outputCount,
    int numDetections,
    float confThreshold,
    cudaStream_t stream)
{
    dim3 block_size(256);
    dim3 grid_size((numDetections + block_size.x - 1) / block_size.x);
    
    filter_detections_kernel<<<grid_size, block_size, 0, stream>>>(
        inputDetections, suppressed, outputDetections, outputCount, 
        numDetections, confThreshold
    );
}

void launch_sort_detections_kernel(
    float* detections,
    int* indices,
    int numDetections,
    cudaStream_t stream)
{
    // Initialize indices
    dim3 block_size(256);
    dim3 grid_size((numDetections + block_size.x - 1) / block_size.x);
    
    prepare_sort_indices_kernel<<<grid_size, block_size, 0, stream>>>(
        indices, numDetections
    );
    
    // Sort indices by confidence using Thrust
    thrust::device_ptr<int> indices_ptr(indices);
    thrust::sort(indices_ptr, indices_ptr + numDetections, 
                ConfidenceComparator(detections));
    
    // Allocate temporary buffer for reordered detections
    float* temp_detections;
    cudaMalloc(&temp_detections, numDetections * 6 * sizeof(float));
    
    // Reorder detections based on sorted indices
    reorder_detections_kernel<<<grid_size, block_size, 0, stream>>>(
        detections, indices, temp_detections, numDetections
    );
    
    // Copy back to original buffer
    cudaMemcpyAsync(detections, temp_detections, 
                   numDetections * 6 * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);
    
    cudaFree(temp_detections);
}

} // extern "C"