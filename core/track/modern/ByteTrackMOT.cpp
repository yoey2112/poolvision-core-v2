#include "ByteTrackMOT.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>

namespace pv {
namespace modern {

// Helper function for Hungarian algorithm (simplified for pool tracking)
namespace {
    class HungarianSolver {
    public:
        static std::vector<std::vector<int>> solve(const std::vector<std::vector<float>>& costMatrix,
                                                  float threshold) {
            std::vector<std::vector<int>> assignments;
            if (costMatrix.empty()) return assignments;
            
            int rows = static_cast<int>(costMatrix.size());
            int cols = static_cast<int>(costMatrix[0].size());
            
            // Simple greedy assignment for high-performance tracking
            // For pool balls, we typically have few objects so this is sufficient
            std::vector<bool> usedRows(rows, false);
            std::vector<bool> usedCols(cols, false);
            
            // Find minimum cost assignments
            while (true) {
                float minCost = threshold + 1.0f;
                int bestRow = -1, bestCol = -1;
                
                for (int i = 0; i < rows; ++i) {
                    if (usedRows[i]) continue;
                    for (int j = 0; j < cols; ++j) {
                        if (usedCols[j]) continue;
                        if (costMatrix[i][j] < minCost) {
                            minCost = costMatrix[i][j];
                            bestRow = i;
                            bestCol = j;
                        }
                    }
                }
                
                if (bestRow == -1 || minCost > threshold) break;
                
                assignments.push_back({bestRow, bestCol});
                usedRows[bestRow] = true;
                usedCols[bestCol] = true;
            }
            
            return assignments;
        }
    };
}

// ByteTrack implementation
ByteTrackMOT::ByteTrack::ByteTrack() 
    : trackId(-1), confidence(0.0f), age(0), timeSinceUpdate(0), hits(0),
      isActivated(false), state(New) {
}

ByteTrackMOT::ByteTrack::ByteTrack(const cv::Rect2f& bbox, float conf, int id)
    : trackId(id), bbox(bbox), confidence(conf), age(0), timeSinceUpdate(0), 
      hits(1), isActivated(false), state(New) {
    velocity = cv::Point2f(0, 0);
    acceleration = cv::Point2f(0, 0);
    initKalmanFilter(bbox);
}

void ByteTrackMOT::ByteTrack::initKalmanFilter(const cv::Rect2f& bbox) {
    // 8-state Kalman filter: [x, y, w, h, vx, vy, vw, vh]
    kalmanFilter.init(8, 4, 0);
    
    // State transition matrix (constant velocity model)
    kalmanFilter.transitionMatrix = (cv::Mat_<float>(8, 8) <<
        1, 0, 0, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1);
    
    // Measurement matrix
    kalmanFilter.measurementMatrix = (cv::Mat_<float>(4, 8) <<
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0);
    
    // Process noise covariance
    cv::setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(1e-2));
    kalmanFilter.processNoiseCov.at<float>(4, 4) = 1e-1; // velocity noise
    kalmanFilter.processNoiseCov.at<float>(5, 5) = 1e-1;
    kalmanFilter.processNoiseCov.at<float>(6, 6) = 1e-3; // size change noise
    kalmanFilter.processNoiseCov.at<float>(7, 7) = 1e-3;
    
    // Measurement noise covariance
    cv::setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(10.0));
    
    // Error covariance
    cv::setIdentity(kalmanFilter.errorCovPost, cv::Scalar::all(100.0));
    
    // Initialize state
    kalmanFilter.statePost.at<float>(0) = bbox.x + bbox.width / 2;
    kalmanFilter.statePost.at<float>(1) = bbox.y + bbox.height / 2;
    kalmanFilter.statePost.at<float>(2) = bbox.width;
    kalmanFilter.statePost.at<float>(3) = bbox.height;
    kalmanFilter.statePost.at<float>(4) = 0; // vx
    kalmanFilter.statePost.at<float>(5) = 0; // vy
    kalmanFilter.statePost.at<float>(6) = 0; // vw
    kalmanFilter.statePost.at<float>(7) = 0; // vh
}

cv::Rect2f ByteTrackMOT::ByteTrack::predict() {
    cv::Mat prediction = kalmanFilter.predict();
    
    float cx = prediction.at<float>(0);
    float cy = prediction.at<float>(1);
    float w = std::max(prediction.at<float>(2), 1.0f);
    float h = std::max(prediction.at<float>(3), 1.0f);
    
    // Update velocity for physics tracking
    velocity.x = prediction.at<float>(4);
    velocity.y = prediction.at<float>(5);
    
    bbox = cv::Rect2f(cx - w/2, cy - h/2, w, h);
    timeSinceUpdate++;
    
    return bbox;
}

void ByteTrackMOT::ByteTrack::update(const cv::Rect2f& newBbox, float conf) {
    confidence = conf;
    timeSinceUpdate = 0;
    hits++;
    
    // Update Kalman filter
    cv::Mat measurement(4, 1, CV_32F);
    measurement.at<float>(0) = newBbox.x + newBbox.width / 2;
    measurement.at<float>(1) = newBbox.y + newBbox.height / 2;
    measurement.at<float>(2) = newBbox.width;
    measurement.at<float>(3) = newBbox.height;
    
    kalmanFilter.correct(measurement);
    
    // Update position history for physics analysis
    cv::Point2f center(newBbox.x + newBbox.width / 2, newBbox.y + newBbox.height / 2);
    positionHistory.push_back(center);
    timestampHistory.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count());
    
    // Keep only recent history
    while (positionHistory.size() > 10) {
        positionHistory.pop_front();
        timestampHistory.pop_front();
    }
    
    // Calculate velocity from history
    if (positionHistory.size() >= 2) {
        const auto& p1 = positionHistory[positionHistory.size()-2];
        const auto& p2 = positionHistory[positionHistory.size()-1];
        double dt = (timestampHistory.back() - timestampHistory[timestampHistory.size()-2]) / 1000.0;
        
        if (dt > 0) {
            velocity.x = static_cast<float>((p2.x - p1.x) / dt);
            velocity.y = static_cast<float>((p2.y - p1.y) / dt);
        }
    }
    
    bbox = newBbox;
}

void ByteTrackMOT::ByteTrack::markMissed() {
    timeSinceUpdate++;
    if (state == Tracked) {
        state = Lost;
    }
}

bool ByteTrackMOT::ByteTrack::isValid() const {
    return bbox.width > 0 && bbox.height > 0 && trackId >= 0;
}

Track ByteTrackMOT::ByteTrack::toTrack() const {
    Track track;
    track.id = trackId;
    track.c = cv::Point2f(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
    track.v = velocity;
    track.r = std::min(bbox.width, bbox.height) / 2; // Approximate radius
    return track;
}

// ByteTrackMOT implementation
ByteTrackMOT::ByteTrackMOT(const Config& config) 
    : config_(config), nextId_(1), frameId_(0) {
}

std::vector<Track> ByteTrackMOT::update(
    const std::vector<pv::DetectionResult::Detection>& detections, double timestamp) {
    
    frameId_++;
    
    // Predict all existing tracks
    predictAllTracks();
    
    // Separate detections by confidence
    auto highDetections = filterDetectionsByConfidence(detections, config_.trackHighThresh);
    auto lowDetections = filterDetectionsByConfidence(detections, config_.trackLowThresh);
    
    // Remove high confidence detections from low confidence set
    std::vector<pv::DetectionResult::Detection> remainingLowDetections;
    for (const auto& lowDet : lowDetections) {
        bool isHigh = false;
        for (const auto& highDet : highDetections) {
            if (calculateIoU(detectionToBBox(lowDet), detectionToBBox(highDet)) > 0.5f) {
                isHigh = true;
                break;
            }
        }
        if (!isHigh) {
            remainingLowDetections.push_back(lowDet);
        }
    }
    
    // Step 1: Associate high confidence detections with tracked tracks
    std::vector<ByteTrack*> trackedTracks;
    for (auto& track : trackedTracks_) {
        if (track.state == ByteTrack::Tracked) {
            trackedTracks.push_back(&track);
        }
    }
    
    auto assignments = linearAssignment(trackedTracks, highDetections, config_.matchThresh);
    
    // Update matched tracks
    for (const auto& assignment : assignments) {
        if (assignment.size() == 2) {
            int trackIdx = assignment[0];
            int detIdx = assignment[1];
            trackedTracks[trackIdx]->update(detectionToBBox(highDetections[detIdx]),
                                          highDetections[detIdx].confidence);
        }
    }
    
    // Collect unmatched tracks and detections
    std::vector<ByteTrack*> unmatchedTracks;
    std::vector<pv::DetectionResult::Detection> unmatchedDetections;
    
    std::vector<bool> matchedTrackFlags(trackedTracks.size(), false);
    std::vector<bool> matchedDetFlags(highDetections.size(), false);
    
    for (const auto& assignment : assignments) {
        if (assignment.size() == 2) {
            matchedTrackFlags[assignment[0]] = true;
            matchedDetFlags[assignment[1]] = true;
        }
    }
    
    for (size_t i = 0; i < trackedTracks.size(); ++i) {
        if (!matchedTrackFlags[i]) {
            unmatchedTracks.push_back(trackedTracks[i]);
        }
    }
    
    for (size_t i = 0; i < highDetections.size(); ++i) {
        if (!matchedDetFlags[i]) {
            unmatchedDetections.push_back(highDetections[i]);
        }
    }
    
    // Step 2: Associate unmatched tracks with low confidence detections
    std::vector<ByteTrack*> lostTracks;
    for (auto& track : lostTracks_) {
        lostTracks.push_back(&track);
    }
    
    auto allUnmatchedTracks = jointStracks(unmatchedTracks, lostTracks);
    auto secondAssignments = linearAssignment(allUnmatchedTracks, remainingLowDetections, 0.5f);
    
    // Update second round matches
    for (const auto& assignment : secondAssignments) {
        if (assignment.size() == 2) {
            int trackIdx = assignment[0];
            int detIdx = assignment[1];
            auto& track = *allUnmatchedTracks[trackIdx];
            track.update(detectionToBBox(remainingLowDetections[detIdx]),
                        remainingLowDetections[detIdx].confidence);
            if (track.state == ByteTrack::Lost) {
                track.state = ByteTrack::Tracked;
            }
        }
    }
    
    // Step 3: Handle remaining unmatched tracks and detections
    std::vector<bool> secondMatchedTrackFlags(allUnmatchedTracks.size(), false);
    std::vector<bool> secondMatchedDetFlags(remainingLowDetections.size(), false);
    
    for (const auto& assignment : secondAssignments) {
        if (assignment.size() == 2) {
            secondMatchedTrackFlags[assignment[0]] = true;
            secondMatchedDetFlags[assignment[1]] = true;
        }
    }
    
    // Mark remaining unmatched tracks as lost
    for (size_t i = 0; i < allUnmatchedTracks.size(); ++i) {
        if (!secondMatchedTrackFlags[i]) {
            allUnmatchedTracks[i]->markMissed();
        }
    }
    
    // Create new tracks from unmatched high confidence detections
    for (size_t i = 0; i < unmatchedDetections.size(); ++i) {
        ByteTrack newTrack(detectionToBBox(unmatchedDetections[i]),
                          unmatchedDetections[i].confidence, nextId_++);
        newTrack.state = ByteTrack::Tracked;
        newTrack.isActivated = true;
        trackedTracks_.push_back(newTrack);
    }
    
    // Update track states and remove old tracks
    updateTrackStates();
    removeOldTracks();
    
    // Convert active tracks to output format
    std::vector<Track> outputTracks;
    for (const auto& track : trackedTracks_) {
        if (track.state == ByteTrack::Tracked && track.isActivated) {
            outputTracks.push_back(track.toTrack());
        }
    }
    
    return outputTracks;
}

std::vector<Track> ByteTrackMOT::updateFromQueue(DetectionQueue& queue) {
    DetectionResult result;
    if (queue.pop(result)) {
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            result.timestamp.time_since_epoch()).count() / 1000.0;
        return update(result.detections, timestamp);
    }
    return {};
}

void ByteTrackMOT::predictAllTracks() {
    for (auto& track : trackedTracks_) {
        track.predict();
    }
    for (auto& track : lostTracks_) {
        track.predict();
    }
}

std::vector<std::vector<int>> ByteTrackMOT::linearAssignment(
    const std::vector<ByteTrack*>& tracks,
    const std::vector<pv::DetectionResult::Detection>& detections,
    float threshold) {
    
    if (tracks.empty() || detections.empty()) {
        return {};
    }
    
    // Build cost matrix
    std::vector<std::vector<float>> costMatrix(tracks.size());
    for (size_t i = 0; i < tracks.size(); ++i) {
        costMatrix[i].resize(detections.size());
        for (size_t j = 0; j < detections.size(); ++j) {
            float iou = calculateIoU(tracks[i]->bbox, detectionToBBox(detections[j]));
            costMatrix[i][j] = 1.0f - iou; // Convert IoU to cost
        }
    }
    
    return HungarianSolver::solve(costMatrix, 1.0f - threshold);
}

float ByteTrackMOT::calculateIoU(const cv::Rect2f& a, const cv::Rect2f& b) const {
    float aArea = a.width * a.height;
    float bArea = b.width * b.height;
    
    if (aArea <= 0 || bArea <= 0) return 0.0f;
    
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    
    float intersectionArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float unionArea = aArea + bArea - intersectionArea;
    
    return unionArea > 0 ? intersectionArea / unionArea : 0.0f;
}

cv::Rect2f ByteTrackMOT::detectionToBBox(const pv::DetectionResult::Detection& det) const {
    return cv::Rect2f(det.x, det.y, det.w, det.h);
}

std::vector<pv::DetectionResult::Detection> ByteTrackMOT::filterDetectionsByConfidence(
    const std::vector<pv::DetectionResult::Detection>& detections, float threshold) const {
    
    std::vector<pv::DetectionResult::Detection> filtered;
    for (const auto& det : detections) {
        if (det.confidence >= threshold) {
            filtered.push_back(det);
        }
    }
    return filtered;
}

std::vector<ByteTrackMOT::ByteTrack*> ByteTrackMOT::jointStracks(
    const std::vector<ByteTrack*>& a, const std::vector<ByteTrack*>& b) {
    
    std::vector<ByteTrack*> result = a;
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

void ByteTrackMOT::updateTrackStates() {
    // Move lost tracks that are too old to removed tracks
    auto it = lostTracks_.begin();
    while (it != lostTracks_.end()) {
        if (it->timeSinceUpdate > config_.trackBuffer) {
            it->state = ByteTrack::Removed;
            removedTracks_.push_back(*it);
            it = lostTracks_.erase(it);
        } else {
            ++it;
        }
    }
    
    // Move tracked tracks that are lost to lost tracks
    it = trackedTracks_.begin();
    while (it != trackedTracks_.end()) {
        if (it->timeSinceUpdate > 1) {
            it->state = ByteTrack::Lost;
            lostTracks_.push_back(*it);
            it = trackedTracks_.erase(it);
        } else {
            ++it;
        }
    }
}

void ByteTrackMOT::removeOldTracks() {
    // Keep removed tracks list bounded
    while (removedTracks_.size() > 100) {
        removedTracks_.erase(removedTracks_.begin());
    }
}

void ByteTrackMOT::reset() {
    trackedTracks_.clear();
    lostTracks_.clear();
    removedTracks_.clear();
    nextId_ = 1;
    frameId_ = 0;
}

int ByteTrackMOT::getActiveTrackCount() const {
    return static_cast<int>(trackedTracks_.size());
}

// TrackingPipelineManager implementation
TrackingPipelineManager::TrackingPipelineManager(ProcessingIsolation& isolation, const Config& config)
    : config_(config), isolation_(&isolation), running_(false) {
    tracker_ = std::make_unique<ByteTrackMOT>(config.byteTrackConfig);
}

TrackingPipelineManager::~TrackingPipelineManager() {
    stop();
}

bool TrackingPipelineManager::start() {
    if (running_.load()) return false;
    
    running_.store(true);
    processingThread_ = std::thread(&TrackingPipelineManager::processingLoop, this);
    return true;
}

void TrackingPipelineManager::stop() {
    running_.store(false);
    if (processingThread_.joinable()) {
        processingThread_.join();
    }
}

std::vector<Track> TrackingPipelineManager::getCurrentTracks() const {
    std::lock_guard<std::mutex> lock(tracksMutex_);
    return currentTracks_;
}

void TrackingPipelineManager::processingLoop() {
    // Set CPU thread affinity for optimal performance
    isolation_->setCpuThreadAffinity();
    
    auto& queue = isolation_->getDetectionQueue();
    
    while (running_.load()) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        auto tracks = tracker_->updateFromQueue(queue);
        
        if (!tracks.empty()) {
            {
                std::lock_guard<std::mutex> lock(tracksMutex_);
                currentTracks_ = std::move(tracks);
            }
            
            framesProcessed_.fetch_add(1);
            
            auto endTime = std::chrono::high_resolution_clock::now();
            double processingTime = std::chrono::duration<double, std::milli>(
                endTime - startTime).count();
            updateMetrics(processingTime);
            isolation_->updateCpuMetrics(processingTime);
        }
        
        // Small sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void TrackingPipelineManager::updateMetrics(double processingTime) {
    if (config_.enableMetrics) {
        double currentAvg = avgProcessingTime_.load();
        double newAvg = currentAvg * 0.9 + processingTime * 0.1; // Exponential moving average
        avgProcessingTime_.store(newAvg);
    }
}

} // namespace modern
} // namespace pv