#include "TensorRtBallDetector.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <numeric>

namespace pv {

TensorRtBallDetector::TensorRtBallDetector()
    : engineReady(false)
    , lastInferenceTime(0.0)
    , avgInferenceTime(0.0)
    , totalInferences(0)
#ifdef USE_TENSORRT
    , d_input(nullptr)
    , d_output(nullptr)
    , stream(nullptr)
#endif
{
#ifdef USE_GPU_ACCELERATION
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    // Initialize preprocessing
    preprocessor = std::make_unique<CudaPreprocessKernels>();
#endif
}

TensorRtBallDetector::~TensorRtBallDetector() {
#ifdef USE_TENSORRT
    deallocateBuffers();
    if (stream) {
        cudaStreamDestroy(stream);
    }
#endif
    
#ifdef USE_GPU_ACCELERATION
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
#endif
}

bool TensorRtBallDetector::initialize(const Config& cfg) {
    config = cfg;
    
#ifndef USE_TENSORRT
    std::cerr << "TensorRT not available in this build" << std::endl;
    return false;
#else
    try {
        // Initialize preprocessing
        CudaPreprocessKernels::PreprocessConfig preprocessConfig;
        preprocessConfig.outputWidth = config.inputWidth;
        preprocessConfig.outputHeight = config.inputHeight;
        preprocessConfig.maintainAspectRatio = true;
        preprocessConfig.useLetterbox = true;
        
        if (!preprocessor->initialize(preprocessConfig)) {
            std::cerr << "Failed to initialize preprocessor" << std::endl;
            return false;
        }
        
        // Create CUDA stream
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream" << std::endl;
            return false;
        }
        
        // Initialize TensorRT logger
        logger = std::make_unique<Logger>();
        
        // Try to load existing engine first, otherwise build from ONNX
        if (!loadEngine()) {
            if (!buildEngine()) {
                std::cerr << "Failed to build TensorRT engine" << std::endl;
                return false;
            }
            saveEngine(); // Cache the built engine
        }
        
        if (!setupBindings()) {
            std::cerr << "Failed to setup tensor bindings" << std::endl;
            return false;
        }
        
        if (!allocateBuffers()) {
            std::cerr << "Failed to allocate GPU buffers" << std::endl;
            return false;
        }
        
        engineReady = true;
        std::cout << "TensorRT ball detector initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "TensorRT initialization error: " << e.what() << std::endl;
        return false;
    }
#endif
}

bool TensorRtBallDetector::loadModel(const std::string& modelPath) {
    config.modelPath = modelPath;
    return buildEngine();
}

std::vector<TensorRtBallDetector::Detection> TensorRtBallDetector::detect(const cv::cuda::GpuMat& image) {
#ifndef USE_TENSORRT
    std::cerr << "TensorRT not available" << std::endl;
    return {};
#else
    if (!engineReady) {
        std::cerr << "Engine not ready" << std::endl;
        return {};
    }
    
    cudaEventRecord(startEvent, stream);
    
    try {
        // Preprocess image
        if (!preprocessor->preprocessImage(image, static_cast<float*>(d_input), stream)) {
            std::cerr << "Preprocessing failed" << std::endl;
            return {};
        }
        
        // Run inference
        if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
            std::cerr << "TensorRT inference failed" << std::endl;
            return {};
        }
        
        // Copy output to host (for post-processing)
        std::vector<float> hostOutput(bindingSizes[1] / sizeof(float));
        if (cudaMemcpyAsync(hostOutput.data(), d_output, bindingSizes[1], 
                           cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            std::cerr << "Failed to copy output to host" << std::endl;
            return {};
        }
        
        cudaStreamSynchronize(stream);
        cudaEventRecord(stopEvent, stream);
        
        // Post-process results
        auto detections = postProcess(hostOutput.data(), hostOutput.size() / 6); // Assuming 6 values per detection
        detections = applyNMS(detections);
        
        // Update performance metrics
        cudaEventSynchronize(stopEvent);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
        lastInferenceTime = static_cast<double>(milliseconds);
        
        totalInferences++;
        avgInferenceTime = (avgInferenceTime * (totalInferences - 1) + lastInferenceTime) / totalInferences;
        
        return detections;
        
    } catch (const std::exception& e) {
        std::cerr << "Detection error: " << e.what() << std::endl;
        return {};
    }
#endif
}

std::vector<Ball> TensorRtBallDetector::detectBalls(const cv::cuda::GpuMat& image) {
    auto detections = detect(image);
    std::vector<Ball> balls;
    balls.reserve(detections.size());
    
    for (const auto& det : detections) {
        balls.push_back(det.toBall());
    }
    
    return balls;
}

#ifdef USE_TENSORRT
bool TensorRtBallDetector::buildEngine() {
    try {
        auto engine_ptr = buildEngineFromOnnx();
        if (!engine_ptr) {
            std::cerr << "Failed to build engine from ONNX" << std::endl;
            return false;
        }
        
        engine = std::move(engine_ptr);
        context.reset(engine->createExecutionContext());
        
        if (!context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Engine building error: " << e.what() << std::endl;
        return false;
    }
}

bool TensorRtBallDetector::loadEngine() {
    try {
        auto engine_ptr = loadEngineFromFile();
        if (!engine_ptr) {
            return false;
        }
        
        engine = std::move(engine_ptr);
        context.reset(engine->createExecutionContext());
        
        return context != nullptr;
        
    } catch (const std::exception& e) {
        std::cerr << "Engine loading error: " << e.what() << std::endl;
        return false;
    }
}

bool TensorRtBallDetector::saveEngine() {
    if (!engine) {
        std::cerr << "No engine to save" << std::endl;
        return false;
    }
    
    try {
        auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(engine->serialize());
        if (!serialized) {
            std::cerr << "Failed to serialize engine" << std::endl;
            return false;
        }
        
        std::ofstream file(config.engineCachePath, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open engine cache file for writing" << std::endl;
            return false;
        }
        
        file.write(static_cast<const char*>(serialized->data()), serialized->size());
        file.close();
        
        std::cout << "Engine saved to " << config.engineCachePath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Engine saving error: " << e.what() << std::endl;
        return false;
    }
}

std::unique_ptr<nvinfer1::ICudaEngine> TensorRtBallDetector::buildEngineFromOnnx() {
    // Create TensorRT builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*logger));
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return nullptr;
    }
    
    // Create network definition
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        std::cerr << "Failed to create network definition" << std::endl;
        return nullptr;
    }
    
    // Create ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *logger));
    if (!parser) {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return nullptr;
    }
    
    // Parse ONNX model
    if (!parser->parseFromFile(config.modelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX model" << std::endl;
        return nullptr;
    }
    
    // Create builder config
    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builderConfig) {
        std::cerr << "Failed to create builder config" << std::endl;
        return nullptr;
    }
    
    // Set workspace size
    builderConfig->setMaxWorkspaceSize(config.workspaceSize);
    
    // Enable optimizations
    if (config.useFP16 && builder->platformHasFastFp16()) {
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 optimization enabled" << std::endl;
    }
    
    // Build and return engine
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *builderConfig));
    if (!engine) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        return nullptr;
    }
    
    std::cout << "TensorRT engine built successfully" << std::endl;
    return engine;
}

std::unique_ptr<nvinfer1::ICudaEngine> TensorRtBallDetector::loadEngineFromFile() {
    std::ifstream file(config.engineCachePath, std::ios::binary);
    if (!file) {
        return nullptr;
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    
    if (!runtime) {
        runtime.reset(nvinfer1::createInferRuntime(*logger));
    }
    
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(buffer.data(), size, nullptr));
    
    if (engine) {
        std::cout << "Engine loaded from " << config.engineCachePath << std::endl;
    }
    
    return engine;
}

bool TensorRtBallDetector::setupBindings() {
    if (!engine) return false;
    
    int numBindings = engine->getNbBindings();
    bindings.resize(numBindings);
    bindingSizes.resize(numBindings);
    
    for (int i = 0; i < numBindings; i++) {
        auto dims = engine->getBindingDimensions(i);
        auto type = engine->getBindingDataType(i);
        
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; j++) {
            size *= dims.d[j];
        }
        
        if (type == nvinfer1::DataType::kFLOAT) {
            size *= sizeof(float);
        } else if (type == nvinfer1::DataType::kHALF) {
            size *= sizeof(uint16_t);
        }
        
        bindingSizes[i] = size;
        
        if (engine->bindingIsInput(i)) {
            inputNames.push_back(engine->getBindingName(i));
            inputDims.push_back(dims);
        } else {
            outputNames.push_back(engine->getBindingName(i));
            outputDims.push_back(dims);
        }
    }
    
    return true;
}

bool TensorRtBallDetector::allocateBuffers() {
    try {
        // Allocate input buffer
        if (cudaMalloc(&d_input, bindingSizes[0]) != cudaSuccess) {
            std::cerr << "Failed to allocate input buffer" << std::endl;
            return false;
        }
        bindings[0] = d_input;
        
        // Allocate output buffer
        if (cudaMalloc(&d_output, bindingSizes[1]) != cudaSuccess) {
            std::cerr << "Failed to allocate output buffer" << std::endl;
            return false;
        }
        bindings[1] = d_output;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Buffer allocation error: " << e.what() << std::endl;
        return false;
    }
}

void TensorRtBallDetector::deallocateBuffers() {
    if (d_input) {
        cudaFree(d_input);
        d_input = nullptr;
    }
    
    if (d_output) {
        cudaFree(d_output);
        d_output = nullptr;
    }
}

std::vector<TensorRtBallDetector::Detection> TensorRtBallDetector::postProcess(float* output, int numDetections) {
    std::vector<Detection> detections;
    detections.reserve(numDetections);
    
    // Assuming YOLO output format: [x, y, w, h, confidence, class_probs...]
    for (int i = 0; i < numDetections; i++) {
        int offset = i * 6; // 6 values per detection
        
        float confidence = output[offset + 4];
        if (confidence < config.confThreshold) {
            continue;
        }
        
        Detection det;
        det.x = output[offset + 0];
        det.y = output[offset + 1];
        det.w = output[offset + 2];
        det.h = output[offset + 3];
        det.confidence = confidence;
        det.classId = static_cast<int>(output[offset + 5]);
        
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<TensorRtBallDetector::Detection> TensorRtBallDetector::applyNMS(const std::vector<Detection>& detections) {
    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);
    
    // Sort by confidence
    std::vector<int> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });
    
    for (int i = 0; i < indices.size(); i++) {
        if (suppressed[i]) continue;
        
        int idx = indices[i];
        result.push_back(detections[idx]);
        
        // Suppress overlapping detections
        for (int j = i + 1; j < indices.size(); j++) {
            if (suppressed[j]) continue;
            
            int idx2 = indices[j];
            
            // Calculate IoU
            float x1 = std::max(detections[idx].x, detections[idx2].x);
            float y1 = std::max(detections[idx].y, detections[idx2].y);
            float x2 = std::min(detections[idx].x + detections[idx].w, 
                               detections[idx2].x + detections[idx2].w);
            float y2 = std::min(detections[idx].y + detections[idx].h,
                               detections[idx2].y + detections[idx2].h);
            
            if (x2 > x1 && y2 > y1) {
                float intersection = (x2 - x1) * (y2 - y1);
                float area1 = detections[idx].w * detections[idx].h;
                float area2 = detections[idx2].w * detections[idx2].h;
                float iou = intersection / (area1 + area2 - intersection);
                
                if (iou > config.nmsThreshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    return result;
}

void TensorRtBallDetector::Logger::log(Severity severity, const char* msg) noexcept {
    switch (severity) {
        case Severity::kERROR:
            std::cerr << "[TensorRT ERROR] " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cout << "[TensorRT WARNING] " << msg << std::endl;
            break;
        case Severity::kINFO:
            std::cout << "[TensorRT INFO] " << msg << std::endl;
            break;
        case Severity::kVERBOSE:
            // Suppress verbose messages
            break;
    }
}

#endif // USE_TENSORRT

} // namespace pv