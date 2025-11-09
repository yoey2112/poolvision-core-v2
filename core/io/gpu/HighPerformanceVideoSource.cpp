#include "HighPerformanceVideoSource.hpp"

#ifdef USE_GPU_ACCELERATION

#include <chrono>
#include <iostream>
#include <algorithm>

namespace pv {

HighPerformanceVideoSource::HighPerformanceVideoSource() 
    : useHardwareAcceleration(true)
    , hardwareReady(false)
    , running(false)
    , captureReady(false)
    , maxBufferSize(DEFAULT_BUFFER_SIZE)
    , targetResolution(1920, 1080)
    , sourceFPS(30.0)
#ifdef USE_NVDEC
    , cuContext(nullptr)
    , cuDevice(0)
    , cudaStream(nullptr)
#endif
{
    lastFrameTime = std::chrono::high_resolution_clock::now();
}

HighPerformanceVideoSource::~HighPerformanceVideoSource() {
    release();
}

bool HighPerformanceVideoSource::open(const std::string &source) {
    sourceString = source;
    
    if (useHardwareAcceleration && initializeNVDEC()) {
        hardwareReady = true;
        std::cout << "Hardware acceleration enabled with NVDEC" << std::endl;
    } else {
        hardwareReady = false;
        if (!initializeFallback()) {
            std::cerr << "Failed to initialize video source" << std::endl;
            return false;
        }
        std::cout << "Using OpenCV fallback for video capture" << std::endl;
    }
    
    // Start processing threads
    running = true;
    captureThread = std::thread(&HighPerformanceVideoSource::captureLoop, this);
    processThread = std::thread(&HighPerformanceVideoSource::processFrame, this);
    
    return true;
}

bool HighPerformanceVideoSource::read(cv::cuda::GpuMat &frame) {
    std::unique_lock<std::mutex> lock(queueMutex);
    
    // Wait for frame with timeout
    auto timeout = std::chrono::milliseconds(100);
    if (!queueCondition.wait_for(lock, timeout, [this] { return !frameQueue.empty() || !running; })) {
        return false;
    }
    
    if (frameQueue.empty() || !running) {
        return false;
    }
    
    auto frameBuffer = frameQueue.front();
    frameQueue.pop();
    lock.unlock();
    
    if (frameBuffer.valid) {
        if (hardwareReady) {
            frame = frameBuffer.gpuFrame;
        } else {
            // Upload to GPU if using fallback
            frameBuffer.cpuFrame.upload(frame);
        }
        
        // Update metrics
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - lastFrameTime).count();
        lastFrameTime = currentTime;
        
        metrics.framesCaptured++;
        metrics.avgCaptureTime = (metrics.avgCaptureTime * 0.9) + (elapsed * 0.1);
        
        return true;
    }
    
    return false;
}

bool HighPerformanceVideoSource::readCPU(cv::Mat &frame) {
    cv::cuda::GpuMat gpuFrame;
    if (read(gpuFrame)) {
        gpuFrame.download(frame);
        return true;
    }
    return false;
}

void HighPerformanceVideoSource::release() {
    running = false;
    queueCondition.notify_all();
    
    if (captureThread.joinable()) {
        captureThread.join();
    }
    
    if (processThread.joinable()) {
        processThread.join();
    }
    
    cleanupResources();
    
    // Clear frame queue
    std::lock_guard<std::mutex> lock(queueMutex);
    while (!frameQueue.empty()) {
        frameQueue.pop();
    }
}

double HighPerformanceVideoSource::fps() const {
    if (hardwareReady) {
        return sourceFPS;
    } else if (fallbackCapture.isOpened()) {
        return fallbackCapture.get(cv::CAP_PROP_FPS);
    }
    return 0.0;
}

void HighPerformanceVideoSource::setUseHardwareAcceleration(bool enable) {
    useHardwareAcceleration = enable;
}

void HighPerformanceVideoSource::captureLoop() {
    auto frameStart = std::chrono::high_resolution_clock::now();
    
    while (running) {
        FrameBuffer buffer;
        buffer.valid = false;
        buffer.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            
        bool captured = false;
        
        if (hardwareReady) {
#ifdef USE_NVDEC
            // NVDEC hardware capture would go here
            // For now, use a placeholder that simulates hardware capture
            captured = false; // Would be: nvDecoder->Decode(buffer.gpuFrame);
#endif
        } else if (fallbackCapture.isOpened()) {
            captured = fallbackCapture.read(buffer.cpuFrame);
        }
        
        if (captured) {
            buffer.valid = true;
            metrics.framesDecoded++;
            
            // Calculate decode time
            auto frameEnd = std::chrono::high_resolution_clock::now();
            auto decodeTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                frameEnd - frameStart).count();
            metrics.avgDecodeTime = (metrics.avgDecodeTime * 0.9) + (decodeTime * 0.1);
            
            // Add to queue
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                
                // Drop old frames if buffer is full
                while (frameQueue.size() >= maxBufferSize && running) {
                    frameQueue.pop();
                    metrics.framesDropped++;
                }
                
                if (running) {
                    frameQueue.push(std::move(buffer));
                    lock.unlock();
                    queueCondition.notify_one();
                }
            }
        }
        
        frameStart = std::chrono::high_resolution_clock::now();
        
        // Control frame rate if needed
        auto sleepTime = std::chrono::microseconds(static_cast<int64_t>(1000000.0 / sourceFPS));
        std::this_thread::sleep_for(sleepTime);
    }
}

void HighPerformanceVideoSource::processFrame() {
    // Additional frame processing could be added here
    // For now, this thread is reserved for future GPU preprocessing
    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

bool HighPerformanceVideoSource::initializeNVDEC() {
#ifdef USE_NVDEC
    try {
        // Initialize CUDA context
        if (cuInit(0) != CUDA_SUCCESS) {
            std::cerr << "Failed to initialize CUDA" << std::endl;
            return false;
        }
        
        if (cuDeviceGet(&cuDevice, 0) != CUDA_SUCCESS) {
            std::cerr << "Failed to get CUDA device" << std::endl;
            return false;
        }
        
        if (cuCtxCreate(&cuContext, 0, cuDevice) != CUDA_SUCCESS) {
            std::cerr << "Failed to create CUDA context" << std::endl;
            return false;
        }
        
        // Create CUDA stream for asynchronous operations
        if (cudaStreamCreate(&cudaStream) != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream" << std::endl;
            return false;
        }
        
        // Initialize NVDEC decoder
        // nvDecoder = std::make_unique<NvDecoder>(cuContext, sourceString);
        
        std::cout << "NVDEC initialization completed" << std::endl;
        return true;
        
    } catch (const std::exception &e) {
        std::cerr << "NVDEC initialization error: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "NVDEC not available in this build" << std::endl;
    return false;
#endif
}

bool HighPerformanceVideoSource::initializeFallback() {
    // Try to open as camera index first
    try {
        int cameraIndex = std::stoi(sourceString);
        if (fallbackCapture.open(cameraIndex)) {
            configureFallbackCapture();
            return true;
        }
    } catch (...) {
        // Not a number, try as file/URL
    }
    
    // Try to open as file or URL
    if (fallbackCapture.open(sourceString)) {
        configureFallbackCapture();
        return true;
    }
    
    return false;
}

void HighPerformanceVideoSource::configureFallbackCapture() {
    // Optimize OpenCV capture settings
    fallbackCapture.set(cv::CAP_PROP_BUFFERSIZE, 1);
    fallbackCapture.set(cv::CAP_PROP_FPS, sourceFPS);
    
    if (targetResolution.width > 0 && targetResolution.height > 0) {
        fallbackCapture.set(cv::CAP_PROP_FRAME_WIDTH, targetResolution.width);
        fallbackCapture.set(cv::CAP_PROP_FRAME_HEIGHT, targetResolution.height);
    }
    
    // Try to enable hardware acceleration in OpenCV
    fallbackCapture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('H', '2', '6', '4'));
    
    sourceFPS = fallbackCapture.get(cv::CAP_PROP_FPS);
    if (sourceFPS <= 0) sourceFPS = 30.0;
}

void HighPerformanceVideoSource::configureFallbackCapture() {
    // Optimize OpenCV capture settings
    fallbackCapture.set(cv::CAP_PROP_BUFFERSIZE, 1);
    fallbackCapture.set(cv::CAP_PROP_FPS, sourceFPS);
    
    if (targetResolution.width > 0 && targetResolution.height > 0) {
        fallbackCapture.set(cv::CAP_PROP_FRAME_WIDTH, targetResolution.width);
        fallbackCapture.set(cv::CAP_PROP_FRAME_HEIGHT, targetResolution.height);
    }
    
    // Try to enable hardware acceleration in OpenCV
    fallbackCapture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('H', '2', '6', '4'));
    
    sourceFPS = fallbackCapture.get(cv::CAP_PROP_FPS);
    if (sourceFPS <= 0) sourceFPS = 30.0;
}

void HighPerformanceVideoSource::cleanupResources() {
#ifdef USE_NVDEC
    if (cudaStream) {
        cudaStreamDestroy(cudaStream);
        cudaStream = nullptr;
    }
    
    if (cuContext) {
        cuCtxDestroy(cuContext);
        cuContext = nullptr;
    }
#endif
    
    if (fallbackCapture.isOpened()) {
        fallbackCapture.release();
    }
}

std::unique_ptr<HighPerformanceVideoSource> createOptimalVideoSource(
    const std::string &source, bool preferHardwareAcceleration) {
    
    auto videoSource = std::make_unique<HighPerformanceVideoSource>();
    videoSource->setUseHardwareAcceleration(preferHardwareAcceleration);
    
    if (videoSource->open(source)) {
        return videoSource;
    }
    
    return nullptr;
}

} // namespace pv

#endif // USE_GPU_ACCELERATION