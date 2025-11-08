#include "VideoSource.hpp"
#include <iostream>

using namespace pv;

VideoSource::VideoSource() : running(false) {}

VideoSource::~VideoSource() {
    release();
}

bool VideoSource::open(const std::string &source) {
    // try integer
    bool success = false;
    try {
        int idx = std::stoi(source);
        success = cap.open(idx);
    } catch(...) {
        success = cap.open(source);
    }
    
    if (success) {
        // Set backend-specific optimizations
        cap.set(cv::CAP_PROP_BUFFERSIZE, BUFFER_SIZE);
        
        // Start background frame buffering
        running = true;
        bufferThread = std::thread(&VideoSource::bufferFrames, this);
    }
    return success;
}

void VideoSource::bufferFrames() {
    cv::Mat frame;
    while (running) {
        if (!cap.isOpened()) break;
        
        if (cap.read(frame)) {
            std::unique_lock<std::mutex> lock(bufferMutex);
            while (running && frameBuffer.size() >= BUFFER_SIZE) {
                bufferCondition.wait(lock);
            }
            if (!running) break;
            
            frameBuffer.push(frame.clone());
            lock.unlock();
            bufferCondition.notify_one();
        }
    }
}

bool VideoSource::read(cv::Mat &frame) {
    if (!cap.isOpened()) return false;
    
    std::unique_lock<std::mutex> lock(bufferMutex);
    while (running && frameBuffer.empty()) {
        bufferCondition.wait(lock);
    }
    
    if (!frameBuffer.empty()) {
        frame = frameBuffer.front();
        frameBuffer.pop();
        lock.unlock();
        bufferCondition.notify_one();
        return true;
    }
    return false;
}

double VideoSource::fps() const {
    if (cap.isOpened()) return cap.get(cv::CAP_PROP_FPS); 
    return 0.0;
}

void VideoSource::release() {
    running = false;
    bufferCondition.notify_all();
    
    if (bufferThread.joinable()) {
        bufferThread.join();
    }
    
    if (cap.isOpened()) {
        cap.release();
    }
    
    std::unique_lock<std::mutex> lock(bufferMutex);
    while (!frameBuffer.empty()) {
        frameBuffer.pop();
    }
}
