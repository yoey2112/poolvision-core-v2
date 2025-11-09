#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <chrono>

namespace pv {

/**
 * High-performance lock-free queue for passing detection results 
 * from GPU inference pipeline to CPU tracking pipeline
 */
template<typename T>
class LockFreeQueue {
public:
    explicit LockFreeQueue(size_t capacity = 1024)
        : capacity_(capacity)
        , mask_(capacity - 1)
        , buffer_(capacity)
        , head_(0)
        , tail_(0)
    {
        // Ensure capacity is power of 2 for efficient modulo operation
        if ((capacity & (capacity - 1)) != 0) {
            throw std::invalid_argument("Capacity must be a power of 2");
        }
    }

    // Producer side (GPU inference thread)
    bool push(const T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & mask_;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Queue is full
        }
        
        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    // Move semantics for better performance
    bool push(T&& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & mask_;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Queue is full
        }
        
        buffer_[current_tail] = std::move(item);
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    // Consumer side (CPU tracking thread)
    bool pop(T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Queue is empty
        }
        
        item = buffer_[current_head];
        head_.store((current_head + 1) & mask_, std::memory_order_release);
        return true;
    }

    // Non-blocking peek
    bool peek(T& item) const {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Queue is empty
        }
        
        item = buffer_[current_head];
        return true;
    }

    // Queue status
    bool empty() const {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    bool full() const {
        const size_t current_tail = tail_.load(std::memory_order_acquire);
        const size_t next_tail = (current_tail + 1) & mask_;
        return next_tail == head_.load(std::memory_order_acquire);
    }

    size_t size() const {
        const size_t current_head = head_.load(std::memory_order_acquire);
        const size_t current_tail = tail_.load(std::memory_order_acquire);
        return (current_tail - current_head) & mask_;
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    const size_t capacity_;
    const size_t mask_;
    std::vector<T> buffer_;
    
    alignas(64) std::atomic<size_t> head_;  // Cache line alignment
    alignas(64) std::atomic<size_t> tail_;
};

/**
 * Timestamped detection result for GPU->CPU pipeline
 */
struct DetectionResult {
    struct Detection {
        float x, y, w, h;
        float confidence;
        int classId;
    };
    
    std::vector<Detection> detections;
    std::chrono::high_resolution_clock::time_point timestamp;
    uint64_t frameId;
    
    DetectionResult() : frameId(0) {
        timestamp = std::chrono::high_resolution_clock::now();
    }
    
    DetectionResult(std::vector<Detection> dets, uint64_t id)
        : detections(std::move(dets)), frameId(id) {
        timestamp = std::chrono::high_resolution_clock::now();
    }
};

/**
 * Specialized lock-free queue for detection results
 */
using DetectionQueue = LockFreeQueue<DetectionResult>;

/**
 * Performance isolation manager for GPU/CPU pipeline separation
 */
class ProcessingIsolation {
public:
    ProcessingIsolation();
    ~ProcessingIsolation();
    
    // Initialize with CPU core affinity settings
    bool initialize(int gpuCores = 4, int cpuCores = 4);
    
    // Set thread affinity for optimal performance
    bool setGpuThreadAffinity();   // GPU inference threads
    bool setCpuThreadAffinity();   // CPU tracking threads
    bool setUIThreadAffinity();    // UI rendering thread
    
    // Queue management
    DetectionQueue& getDetectionQueue() { return detectionQueue_; }
    
    // Performance monitoring
    struct Metrics {
        std::atomic<uint64_t> gpuFramesProcessed{0};
        std::atomic<uint64_t> cpuFramesProcessed{0};
        std::atomic<uint64_t> queueOverflows{0};
        std::atomic<double> avgGpuLatency{0.0};
        std::atomic<double> avgCpuLatency{0.0};
    };
    
    const Metrics& getMetrics() const { return metrics_; }
    void updateGpuMetrics(double latency);
    void updateCpuMetrics(double latency);

private:
    DetectionQueue detectionQueue_;
    Metrics metrics_;
    
    // CPU affinity settings
    int gpuCoreCount_;
    int cpuCoreCount_;
    std::vector<int> gpuCoreIds_;
    std::vector<int> cpuCoreIds_;
    std::vector<int> uiCoreIds_;
    
    bool initialized_;
    
    // Platform-specific thread affinity helpers
    bool setThreadAffinity(const std::vector<int>& coreIds);
    std::vector<int> getAvailableCores();
};

} // namespace pv