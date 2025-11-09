#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include "../../performance/ProcessingIsolation.hpp"
#include "../../util/Types.hpp"
#include "../Physics.hpp"

namespace pv {
namespace modern {

/**
 * ByteTrack: Multi-Object Tracking by Associating Every Detection Box
 * High-performance CPU tracking optimized for pool ball tracking
 * 
 * Based on: https://arxiv.org/abs/2110.06864
 * Optimized for 300+ FPS performance on CPU
 */
class ByteTrackMOT {
public:
    struct Config {
        // Detection thresholds
        float trackHighThresh = 0.6f;    // High confidence threshold for primary tracking
        float trackLowThresh = 0.1f;     // Low confidence threshold for recovery
        float matchThresh = 0.8f;        // IoU threshold for matching
        
        // Tracking parameters
        int frameRate = 60;              // Expected frame rate
        int trackBuffer = 30;            // Buffer size for lost tracks
        float lambda = 0.98f;            // Exponential moving average factor
        
        // Physics constraints for pool balls
        float maxVelocity = 2000.0f;     // Max pixel velocity per second
        float maxAcceleration = 5000.0f; // Max acceleration
        bool usePhysicsConstraints = true;
    };
    
    struct ByteTrack {
        int trackId;
        cv::Rect2f bbox;
        cv::Point2f velocity;
        cv::Point2f acceleration;
        float confidence;
        int age;                         // Total frames tracked
        int timeSinceUpdate;            // Frames since last update
        int hits;                       // Consecutive successful matches
        bool isActivated;               // Track is confirmed and active
        
        enum State {
            New,        // Just created, not confirmed
            Tracked,    // Active tracking
            Lost,       // Temporarily lost but recoverable
            Removed     // Permanently removed
        } state;
        
        // Kalman filter for prediction
        cv::KalmanFilter kalmanFilter;
        
        // Physics state for pool balls
        std::deque<cv::Point2f> positionHistory;
        std::deque<double> timestampHistory;
        
        ByteTrack();
        ByteTrack(const cv::Rect2f& bbox, float conf, int id);
        
        void initKalmanFilter(const cv::Rect2f& bbox);
        cv::Rect2f predict();
        void update(const cv::Rect2f& bbox, float conf);
        void markMissed();
        bool isValid() const;
        
        // Convert to Pool Vision Track format
        Track toTrack() const;
    };

private:
    Config config_;
    int nextId_;
    int frameId_;
    
    std::vector<ByteTrack> trackedTracks_;   // Active high-confidence tracks
    std::vector<ByteTrack> lostTracks_;      // Temporarily lost tracks
    std::vector<ByteTrack> removedTracks_;   // Recently removed tracks (for debugging)
    
    // Performance optimization
    std::vector<cv::Rect2f> trackPredictions_;
    std::vector<std::vector<float>> distanceMatrix_;
    std::vector<int> assignment_;

public:
    explicit ByteTrackMOT(const Config& config = Config{});
    ~ByteTrackMOT() = default;
    
    // Main tracking interface
    std::vector<Track> update(const std::vector<pv::DetectionResult::Detection>& detections,
                             double timestamp);
    
    // Batch processing for performance
    std::vector<Track> updateFromQueue(DetectionQueue& queue);
    
    // Configuration
    void setConfig(const Config& config) { config_ = config; }
    const Config& getConfig() const { return config_; }
    
    // State management
    void reset();
    int getActiveTrackCount() const;
    int getTotalTrackCount() const { return nextId_ - 1; }
    
    // Debugging and analysis
    std::vector<ByteTrack> getAllTracks() const;
    void getTrackingStatistics(int& active, int& lost, int& total) const;

private:
    // Core ByteTrack algorithm steps
    std::vector<ByteTrack*> jointStracks(const std::vector<ByteTrack*>& a,
                                        const std::vector<ByteTrack*>& b);
    std::vector<ByteTrack*> subStracks(const std::vector<ByteTrack*>& a,
                                      const std::vector<ByteTrack*>& b);
    void removeDuplicateStracks(std::vector<ByteTrack*>& resa,
                               std::vector<ByteTrack*>& resb,
                               const std::vector<ByteTrack*>& stracksa,
                               const std::vector<ByteTrack*>& stracksb);
    
    // Assignment algorithms
    std::vector<std::vector<int>> linearAssignment(
        const std::vector<ByteTrack*>& tracks,
        const std::vector<pv::DetectionResult::Detection>& detections,
        float threshold);
    
    float calculateIoU(const cv::Rect2f& a, const cv::Rect2f& b) const;
    float calculateDistance(const ByteTrack& track, 
                          const pv::DetectionResult::Detection& detection) const;
    
    // Prediction and physics
    void predictAllTracks();
    bool validatePhysics(const ByteTrack& track, 
                        const pv::DetectionResult::Detection& detection,
                        double timestamp) const;
    
    // Track lifecycle management
    void activateTrack(ByteTrack& track, int frameId);
    void reactivateTrack(ByteTrack& track, const pv::DetectionResult::Detection& det, int frameId);
    void loseTrack(ByteTrack& track);
    void removeTrack(ByteTrack& track);
    
    // Cleanup and maintenance
    void removeOldTracks();
    void updateTrackStates();
    
    // Utility functions
    cv::Rect2f detectionToBBox(const pv::DetectionResult::Detection& det) const;
    std::vector<pv::DetectionResult::Detection> filterDetectionsByConfidence(
        const std::vector<pv::DetectionResult::Detection>& detections,
        float threshold) const;
};

/**
 * High-level tracking pipeline manager that connects ByteTrack 
 * to Agent Group 1 GPU pipeline via ProcessingIsolation
 */
class TrackingPipelineManager {
public:
    struct Config {
        ByteTrackMOT::Config byteTrackConfig;
        int maxQueueSize = 100;
        double timeoutMs = 1000.0;
        bool enableMetrics = true;
    };

private:
    Config config_;
    std::unique_ptr<ByteTrackMOT> tracker_;
    ProcessingIsolation* isolation_;
    
    // Processing thread management
    std::atomic<bool> running_;
    std::thread processingThread_;
    
    // Output for Agent Group 3
    std::vector<Track> currentTracks_;
    mutable std::mutex tracksMutex_;
    
    // Performance metrics
    std::atomic<uint64_t> framesProcessed_{0};
    std::atomic<double> avgProcessingTime_{0.0};

public:
    explicit TrackingPipelineManager(ProcessingIsolation& isolation,
                                   const Config& config = Config{});
    ~TrackingPipelineManager();
    
    // Lifecycle
    bool start();
    void stop();
    bool isRunning() const { return running_.load(); }
    
    // Access current tracking results (for Agent Group 3)
    std::vector<Track> getCurrentTracks() const;
    
    // Configuration
    void updateConfig(const Config& config);
    
    // Performance monitoring
    uint64_t getFramesProcessed() const { return framesProcessed_.load(); }
    double getAvgProcessingTime() const { return avgProcessingTime_.load(); }

private:
    void processingLoop();
    void updateMetrics(double processingTime);
};

} // namespace modern
} // namespace pv