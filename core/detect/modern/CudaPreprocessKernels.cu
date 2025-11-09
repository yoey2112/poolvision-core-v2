#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Optimized CUDA kernel for combined resize, letterbox, normalization and format conversion
__global__ void resize_letterbox_normalize_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
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
    float std_b)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tx >= output_width || ty >= output_height) return;
    
    int output_idx = ty * output_width + tx;
    
    // Check if we're in the letterbox area (padding)
    if (tx < offset_x || tx >= offset_x + (int)(input_width * scale_x) ||
        ty < offset_y || ty >= offset_y + (int)(input_height * scale_y)) {
        
        // Padding area - set to normalized gray value
        float gray_norm = (114.0f - 128.0f) / 255.0f; // Normalized gray
        output[output_idx] = gray_norm;  // R
        output[output_idx + output_width * output_height] = gray_norm;  // G
        output[output_idx + 2 * output_width * output_height] = gray_norm;  // B
        return;
    }
    
    // Map to input coordinates
    float src_x = (tx - offset_x) / scale_x;
    float src_y = (ty - offset_y) / scale_y;
    
    // Bilinear interpolation
    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = min(x0 + 1, input_width - 1);
    int y1 = min(y0 + 1, input_height - 1);
    
    x0 = max(0, x0);
    y0 = max(0, y0);
    
    float dx = src_x - x0;
    float dy = src_y - y0;
    
    // Sample four neighboring pixels
    int idx00 = (y0 * input_width + x0) * 3;
    int idx01 = (y0 * input_width + x1) * 3;
    int idx10 = (y1 * input_width + x0) * 3;
    int idx11 = (y1 * input_width + x1) * 3;
    
    // Interpolate each channel (BGR format in OpenCV)
    for (int c = 0; c < 3; c++) {
        float val00 = input[idx00 + c];
        float val01 = input[idx01 + c];
        float val10 = input[idx10 + c];
        float val11 = input[idx11 + c];
        
        float val_top = val00 * (1.0f - dx) + val01 * dx;
        float val_bottom = val10 * (1.0f - dx) + val11 * dx;
        float interpolated = val_top * (1.0f - dy) + val_bottom * dy;
        
        // Normalize and convert to CHW format
        float normalized;
        if (c == 0) { // B -> R (reverse for RGB)
            normalized = (interpolated / 255.0f - mean_r) / std_r;
            output[output_idx + 2 * output_width * output_height] = normalized;
        } else if (c == 1) { // G -> G
            normalized = (interpolated / 255.0f - mean_g) / std_g;
            output[output_idx + output_width * output_height] = normalized;
        } else { // R -> B (reverse for RGB)
            normalized = (interpolated / 255.0f - mean_b) / std_b;
            output[output_idx] = normalized;
        }
    }
}

// BGR to RGB conversion kernel
__global__ void bgr_to_rgb_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width,
    int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    int pixel_idx = idx * 3;
    
    // Swap B and R channels
    output[pixel_idx] = input[pixel_idx + 2];     // R
    output[pixel_idx + 1] = input[pixel_idx + 1]; // G
    output[pixel_idx + 2] = input[pixel_idx];     // B
}

// Normalization kernel
__global__ void normalize_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    int channels,
    float mean_r,
    float mean_g,
    float mean_b,
    float std_r,
    float std_g,
    float std_b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    int pixel_idx = idx * channels;
    
    for (int c = 0; c < channels; c++) {
        float pixel_val = input[pixel_idx + c] / 255.0f;
        float normalized;
        
        if (c == 0) { // B channel
            normalized = (pixel_val - mean_b) / std_b;
        } else if (c == 1) { // G channel
            normalized = (pixel_val - mean_g) / std_g;
        } else { // R channel
            normalized = (pixel_val - mean_r) / std_r;
        }
        
        output[idx + c * total_pixels] = normalized; // CHW format
    }
}

// HWC to CHW format conversion kernel
__global__ void hwc_to_chw_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    int y = idx / width;
    int x = idx % width;
    int hwc_idx = y * width * channels + x * channels;
    
    for (int c = 0; c < channels; c++) {
        int chw_idx = c * total_pixels + y * width + x;
        output[chw_idx] = input[hwc_idx + c];
    }
}

// Host wrapper functions
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
    cudaStream_t stream)
{
    dim3 block_size(16, 16);
    dim3 grid_size(
        (output_width + block_size.x - 1) / block_size.x,
        (output_height + block_size.y - 1) / block_size.y
    );
    
    resize_letterbox_normalize_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, input_width, input_height, output_width, output_height,
        scale_x, scale_y, offset_x, offset_y, mean_r, mean_g, mean_b, std_r, std_g, std_b
    );
}

void launch_bgr_to_rgb_kernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    cudaStream_t stream)
{
    int total_pixels = width * height;
    dim3 block_size(256);
    dim3 grid_size((total_pixels + block_size.x - 1) / block_size.x);
    
    bgr_to_rgb_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, width, height
    );
}

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
    cudaStream_t stream)
{
    int total_pixels = width * height;
    dim3 block_size(256);
    dim3 grid_size((total_pixels + block_size.x - 1) / block_size.x);
    
    normalize_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, width, height, channels, mean_r, mean_g, mean_b, std_r, std_g, std_b
    );
}

void launch_hwc_to_chw_kernel(
    const float* input,
    float* output,
    int width,
    int height,
    int channels,
    cudaStream_t stream)
{
    int total_pixels = width * height;
    dim3 block_size(256);
    dim3 grid_size((total_pixels + block_size.x - 1) / block_size.x);
    
    hwc_to_chw_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, width, height, channels
    );
}

} // extern "C"