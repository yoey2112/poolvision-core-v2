#include "DlDetector.hpp"
#include <iostream>
#include <algorithm>

using namespace pv;

DlDetector::DlDetector()
#ifdef USE_DL_ENGINE
    : env(ORT_LOGGING_LEVEL_WARNING, "poolvision")
#endif
{ 
    ready = false; 
}

DlDetector::~DlDetector(){}

bool DlDetector::loadModel(const std::string &path){
#ifdef USE_DL_ENGINE
    try {
        Ort::SessionOptions sessionOptions;
        // Use all available CPU threads for inference
        sessionOptions.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        sessionOptions.SetInterOpNumThreads(std::thread::hardware_concurrency());
        
        // Enable CUDA if available
        #ifdef USE_CUDA
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        #endif
        
        // Enable all optimizations
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        
        // Enable memory pattern optimization
        sessionOptions.EnableMemPattern();
        sessionOptions.EnableCpuMemArena();
        
        session = std::make_unique<Ort::Session>(env, path.c_str(), sessionOptions);
        
        Ort::AllocatorWithDefaultOptions allocator;
        inputNames = {session->GetInputName(0, allocator)};
        outputNames = {session->GetOutputName(0, allocator)};
        
        ready = true;
        return true;
    } catch(const Ort::Exception &e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    }
#else
    (void)path;
    std::cerr << "DL engine not available: build with -DBUILD_DL_ENGINE=ON and provide ONNX Runtime\n";
    return false;
#endif
}

#ifdef USE_DL_ENGINE
std::vector<Ort::Value> DlDetector::processImage(const cv::Mat &img) {
    static cv::UMat resized;  // Reuse UMat buffer
    cv::resize(img, resized, inputSize, 0, 0, cv::INTER_LINEAR);
    
    // Normalize and convert to float32 using OpenCV's optimized operations
    static cv::UMat float_img;  // Reuse UMat buffer
    resized.convertTo(float_img, CV_32F, 1.0/255.0);
    
    // Pre-allocate input tensor memory
    static std::vector<float> input_tensor(inputSize.width * inputSize.height * 3);
    float *input_ptr = input_tensor.data();
    
    // Optimize HWC to CHW conversion with OpenMP
    cv::Mat float_img_cpu = float_img.getMat(cv::ACCESS_READ);
    #pragma omp parallel for collapse(2)
    for(int c = 0; c < 3; c++) {
        for(int h = 0; h < inputSize.height; h++) {
            float* row_ptr = float_img_cpu.ptr<float>(h);
            for(int w = 0; w < inputSize.width; w++) {
                input_ptr[c * inputSize.width * inputSize.height + h * inputSize.width + w] = 
                    row_ptr[w * 3 + c];
            }
        }
    }
    
    std::vector<int64_t> input_shape = {1, 3, (int64_t)inputSize.height, (int64_t)inputSize.width};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor.size(),
        input_shape.data(), input_shape.size()));
        
    return session->Run(Ort::RunOptions{nullptr}, 
        inputNames.data(), input_tensors.data(), inputNames.size(),
        outputNames.data(), outputNames.size());
}
#endif

std::vector<Ball> DlDetector::detect(const cv::Mat &rectified){
#ifdef USE_DL_ENGINE
    if(!ready || !session) return {};
    
    try {
        auto output_tensors = processImage(rectified);
        if(output_tensors.empty()) return {};
        
        // Get pointer to output tensor
        float* output = output_tensors[0].GetTensorMutableData<float>();
        
        // Process detections (assuming YOLO-style output format)
        std::vector<Ball> detections;
        const int dimensions = 6; // x,y,w,h,conf,class
        const int num_boxes = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1];
        
        float scale_x = (float)rectified.cols / inputSize.width;
        float scale_y = (float)rectified.rows / inputSize.height;
        
        for(int i = 0; i < num_boxes; i++) {
            float confidence = output[i * dimensions + 4];
            if(confidence < confThreshold) continue;
            
            float x = output[i * dimensions + 0] * scale_x;
            float y = output[i * dimensions + 1] * scale_y;
            float w = output[i * dimensions + 2] * scale_x;
            float h = output[i * dimensions + 3] * scale_y;
            int class_id = static_cast<int>(output[i * dimensions + 5]);
            
            Ball b;
            b.c = cv::Point2f(x, y);
            b.r = (w + h) / 4.0f; // Average of width and height / 2
            b.label = class_id;
            b.stripeScore = 0.0f; // Could be determined from model output if trained for it
            detections.push_back(b);
        }
        
        return detections;
    } catch(const Ort::Exception &e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return {};
    }
#else
    (void)rectified;
    return {};
#endif
}
