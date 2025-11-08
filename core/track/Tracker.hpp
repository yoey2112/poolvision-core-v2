#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include "../util/Types.hpp"

namespace pv {

class Tracker {
public:
    struct Params {
        int maxMissingFrames = 30;       // Max frames to keep lost tracks
        float maxMatchDist = 100.0f;     // Max distance for track-detection matching
        int minHits = 3;                 // Min detections before track is confirmed
        int maxAge = 60;                 // Max age for track history
        float processNoise = 1e-2f;      // Process noise for KF
        float measurementNoise = 1e-1f;  // Measurement noise for KF
        float accelNoise = 1e-3f;        // Acceleration noise for KF
    };

    Tracker();
    void setTableSize(const cv::Size &s);
    void setParams(const Params &p) { params = p; }
    void update(const std::vector<Ball> &detections, double timestamp);
    std::vector<Track> tracks() const;
    const std::vector<Track>& confirmedTracks() const { return outTracks; }
    
    // Ball access methods
    const std::vector<Track>& getBalls() const { return outTracks; }
    const Track* getBall(int id) const {
        for(const auto& t : outTracks) {
            if(t.id == id) return &t;
        }
        return nullptr;
    }

private:
    Params params;
    int nextId = 1;
    cv::Size tableSize{1280,720};
    
    struct TrackHistory {
        std::deque<cv::Point2f> positions;
        std::deque<double> timestamps;
        void add(const cv::Point2f &pos, double ts);
        void prune(double maxAge);
    };
    
    struct InternalTrack { 
        int id;
        cv::KalmanFilter kf;
        int missing = 0;
        int hits = 0;
        float r = 0;
        bool confirmed = false;
        TrackHistory history;
        double lastUpdateTime = 0;
    };
    
    std::vector<InternalTrack> its;
    std::vector<Track> outTracks;
    double lastTimestamp = 0;

    // Track management
    void initializeTrack(const Ball &detection);
    void updateTrack(InternalTrack &track, const Ball &detection, double timestamp);
    void predictAll(double timestamp);
    void removeOldTracks();
    void updateTrackStates();
    
    // Assignment
    std::vector<std::vector<double>> costMatrix(const std::vector<cv::Point2f>&pred, const std::vector<cv::Point2f>&meas);
    std::vector<int> hungarianSolve(const std::vector<std::vector<double>>&cost);
    
    // Helper functions
    cv::KalmanFilter createKalmanFilter(const cv::Point2f &pos) const;
};

}
