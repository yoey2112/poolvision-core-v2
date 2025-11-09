#include "ProcessingIsolation.hpp"
#include <iostream>
#include <algorithm>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <processthreadsapi.h>
#elif defined(__linux__)
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#endif

namespace pv {

ProcessingIsolation::ProcessingIsolation()
    : detectionQueue_(1024)  // Power of 2 capacity
    , gpuCoreCount_(0)
    , cpuCoreCount_(0)
    , initialized_(false)
{
}

ProcessingIsolation::~ProcessingIsolation() {
    // Cleanup resources if needed
}

bool ProcessingIsolation::initialize(int gpuCores, int cpuCores) {
    gpuCoreCount_ = gpuCores;
    cpuCoreCount_ = cpuCores;
    
    auto availableCores = getAvailableCores();
    int totalCores = static_cast<int>(availableCores.size());
    
    if (totalCores < (gpuCores + cpuCores + 1)) {
        std::cerr << "Warning: Not enough CPU cores for optimal isolation. "
                  << "Available: " << totalCores 
                  << ", Requested: " << (gpuCores + cpuCores + 1) << std::endl;
    }
    
    // Assign cores with priority: GPU cores (high performance), CPU cores, UI cores
    gpuCoreIds_.clear();
    cpuCoreIds_.clear();
    uiCoreIds_.clear();
    
    int coreIndex = 0;
    
    // Assign GPU cores (use the first high-performance cores)
    for (int i = 0; i < gpuCores && coreIndex < totalCores; i++) {
        gpuCoreIds_.push_back(availableCores[coreIndex++]);
    }
    
    // Assign CPU cores
    for (int i = 0; i < cpuCores && coreIndex < totalCores; i++) {
        cpuCoreIds_.push_back(availableCores[coreIndex++]);
    }
    
    // Assign UI cores (remaining cores)
    while (coreIndex < totalCores) {
        uiCoreIds_.push_back(availableCores[coreIndex++]);
    }
    
    // Ensure we have at least one core for UI
    if (uiCoreIds_.empty() && !cpuCoreIds_.empty()) {
        uiCoreIds_.push_back(cpuCoreIds_.back());
        cpuCoreIds_.pop_back();
    }
    
    initialized_ = true;
    
    std::cout << "Processing isolation initialized:" << std::endl;
    std::cout << "  GPU cores: ";
    for (int id : gpuCoreIds_) std::cout << id << " ";
    std::cout << std::endl;
    std::cout << "  CPU cores: ";
    for (int id : cpuCoreIds_) std::cout << id << " ";
    std::cout << std::endl;
    std::cout << "  UI cores: ";
    for (int id : uiCoreIds_) std::cout << id << " ";
    std::cout << std::endl;
    
    return true;
}

bool ProcessingIsolation::setGpuThreadAffinity() {
    if (!initialized_ || gpuCoreIds_.empty()) {
        std::cerr << "GPU cores not available for affinity setting" << std::endl;
        return false;
    }
    return setThreadAffinity(gpuCoreIds_);
}

bool ProcessingIsolation::setCpuThreadAffinity() {
    if (!initialized_ || cpuCoreIds_.empty()) {
        std::cerr << "CPU cores not available for affinity setting" << std::endl;
        return false;
    }
    return setThreadAffinity(cpuCoreIds_);
}

bool ProcessingIsolation::setUIThreadAffinity() {
    if (!initialized_ || uiCoreIds_.empty()) {
        std::cerr << "UI cores not available for affinity setting" << std::endl;
        return false;
    }
    return setThreadAffinity(uiCoreIds_);
}

void ProcessingIsolation::updateGpuMetrics(double latency) {
    metrics_.gpuFramesProcessed++;
    double currentAvg = metrics_.avgGpuLatency.load();
    uint64_t count = metrics_.gpuFramesProcessed.load();
    double newAvg = (currentAvg * (count - 1) + latency) / count;
    metrics_.avgGpuLatency.store(newAvg);
}

void ProcessingIsolation::updateCpuMetrics(double latency) {
    metrics_.cpuFramesProcessed++;
    double currentAvg = metrics_.avgCpuLatency.load();
    uint64_t count = metrics_.cpuFramesProcessed.load();
    double newAvg = (currentAvg * (count - 1) + latency) / count;
    metrics_.avgCpuLatency.store(newAvg);
}

bool ProcessingIsolation::setThreadAffinity(const std::vector<int>& coreIds) {
    if (coreIds.empty()) {
        return false;
    }

#ifdef _WIN32
    // Windows implementation
    DWORD_PTR affinityMask = 0;
    for (int coreId : coreIds) {
        affinityMask |= (1ULL << coreId);
    }
    
    HANDLE currentThread = GetCurrentThread();
    DWORD_PTR result = SetThreadAffinityMask(currentThread, affinityMask);
    
    if (result == 0) {
        std::cerr << "Failed to set thread affinity: " << GetLastError() << std::endl;
        return false;
    }
    
    // Set thread priority to high for GPU threads, normal for others
    int priority = (coreIds == gpuCoreIds_) ? THREAD_PRIORITY_ABOVE_NORMAL : THREAD_PRIORITY_NORMAL;
    if (!SetThreadPriority(currentThread, priority)) {
        std::cerr << "Failed to set thread priority: " << GetLastError() << std::endl;
    }
    
    return true;
    
#elif defined(__linux__)
    // Linux implementation
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    
    for (int coreId : coreIds) {
        CPU_SET(coreId, &cpuSet);
    }
    
    pthread_t currentThread = pthread_self();
    int result = pthread_setaffinity_np(currentThread, sizeof(cpu_set_t), &cpuSet);
    
    if (result != 0) {
        std::cerr << "Failed to set thread affinity: " << result << std::endl;
        return false;
    }
    
    // Set thread scheduling policy for GPU threads
    if (coreIds == gpuCoreIds_) {
        struct sched_param param;
        param.sched_priority = sched_get_priority_max(SCHED_FIFO);
        
        if (pthread_setschedparam(currentThread, SCHED_FIFO, &param) != 0) {
            // Fall back to normal priority if real-time scheduling fails
            param.sched_priority = 0;
            pthread_setschedparam(currentThread, SCHED_OTHER, &param);
        }
    }
    
    return true;
    
#else
    std::cerr << "Thread affinity not supported on this platform" << std::endl;
    return false;
#endif
}

std::vector<int> ProcessingIsolation::getAvailableCores() {
    std::vector<int> cores;
    
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    int numCores = static_cast<int>(sysInfo.dwNumberOfProcessors);
    
    for (int i = 0; i < numCores; i++) {
        cores.push_back(i);
    }
    
#elif defined(__linux__)
    int numCores = sysconf(_SC_NPROCESSORS_ONLN);
    
    for (int i = 0; i < numCores; i++) {
        cores.push_back(i);
    }
    
#else
    // Fallback: use hardware concurrency
    int numCores = static_cast<int>(std::thread::hardware_concurrency());
    for (int i = 0; i < numCores; i++) {
        cores.push_back(i);
    }
#endif
    
    return cores;
}

} // namespace pv