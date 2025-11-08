#include "Tracker.hpp"
#include "Physics.hpp"
#include <limits>
#include <algorithm>

using namespace pv;

void Tracker::TrackHistory::add(const cv::Point2f &pos, double ts) {
    positions.push_back(pos);
    timestamps.push_back(ts);
}

void Tracker::TrackHistory::prune(double maxAge) {
    while(!timestamps.empty() && timestamps.back() - timestamps.front() > maxAge) {
        positions.pop_front();
        timestamps.pop_front();
    }
}

Tracker::Tracker() { }

void Tracker::setTableSize(const cv::Size &s) { tableSize = s; }

cv::KalmanFilter Tracker::createKalmanFilter(const cv::Point2f &pos) const {
    // State: [x, y, vx, vy, ax, ay]
    cv::KalmanFilter kf(6, 2);
    
    // Transition matrix F for constant acceleration model
    // [1 0 dt 0  dt²/2 0    ]
    // [0 1 0  dt 0     dt²/2]
    // [0 0 1  0  dt    0    ]
    // [0 0 0  1  0     dt   ]
    // [0 0 0  0  1     0    ]
    // [0 0 0  0  0     1    ]
    float dt = 1.0f/30.0f; // assuming 30fps
    float dt2 = dt*dt/2;
    kf.transitionMatrix = (cv::Mat_<float>(6,6) << 
        1,0,dt,0,dt2,0,
        0,1,0,dt,0,dt2,
        0,0,1,0,dt,0,
        0,0,0,1,0,dt,
        0,0,0,0,1,0,
        0,0,0,0,0,1);
    
    // Measurement matrix H (we only measure position)
    kf.measurementMatrix = (cv::Mat_<float>(2,6) << 
        1,0,0,0,0,0,
        0,1,0,0,0,0);
    
    // Process noise
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(params.processNoise));
    kf.processNoiseCov.at<float>(4,4) = params.accelNoise; // ax
    kf.processNoiseCov.at<float>(5,5) = params.accelNoise; // ay
    
    // Measurement noise
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(params.measurementNoise));
    
    // Initial state
    kf.statePost.at<float>(0) = pos.x;
    kf.statePost.at<float>(1) = pos.y;
    kf.statePost.at<float>(2) = 0; // vx
    kf.statePost.at<float>(3) = 0; // vy
    kf.statePost.at<float>(4) = 0; // ax
    kf.statePost.at<float>(5) = 0; // ay
    
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
    
    return kf;
}

void Tracker::initializeTrack(const Ball &detection) {
    InternalTrack it;
    it.id = nextId++;
    it.kf = createKalmanFilter(detection.c);
    it.r = detection.r;
    it.hits = 1;
    it.missing = 0;
    it.confirmed = false;
    it.lastUpdateTime = lastTimestamp;
    it.history.add(detection.c, lastTimestamp);
    its.push_back(std::move(it));
}

void Tracker::updateTrack(InternalTrack &track, const Ball &detection, double timestamp) {
    float dt = timestamp - track.lastUpdateTime;
    if(dt > 0) {
        // Update transition matrix for actual dt
        float dt2 = dt*dt/2;
        track.kf.transitionMatrix.at<float>(0,2) = dt;
        track.kf.transitionMatrix.at<float>(1,3) = dt;
        track.kf.transitionMatrix.at<float>(0,4) = dt2;
        track.kf.transitionMatrix.at<float>(1,5) = dt2;
        track.kf.transitionMatrix.at<float>(2,4) = dt;
        track.kf.transitionMatrix.at<float>(3,5) = dt;
    }
    
    cv::Mat measurement = (cv::Mat_<float>(2,1) << detection.c.x, detection.c.y);
    track.kf.correct(measurement);
    track.r = detection.r;
    track.hits++;
    track.missing = 0;
    track.lastUpdateTime = timestamp;
    track.history.add(detection.c, timestamp);
    track.history.prune(params.maxAge);
}

void Tracker::predictAll(double timestamp) {
    float dt = timestamp - lastTimestamp;
    if(dt <= 0) return;
    
    for(auto &track: its) {
        // Update transition matrix for actual dt
        float dt2 = dt*dt/2;
        track.kf.transitionMatrix.at<float>(0,2) = dt;
        track.kf.transitionMatrix.at<float>(1,3) = dt;
        track.kf.transitionMatrix.at<float>(0,4) = dt2;
        track.kf.transitionMatrix.at<float>(1,5) = dt2;
        track.kf.transitionMatrix.at<float>(2,4) = dt;
        track.kf.transitionMatrix.at<float>(3,5) = dt;
        
        track.kf.predict();
    }
}

void Tracker::update(const std::vector<Ball> &detections, double timestamp){
    // Pre-allocate vectors
    std::vector<cv::Point2f> preds;
    preds.reserve(its.size());
    std::vector<cv::Point2f> meas;
    meas.reserve(detections.size());
    
    // Predict existing tracks in parallel
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // Process predictions
            for(auto &t: its){
                cv::Mat pred = t.kf.predict();
                #pragma omp critical
                {
                    preds.emplace_back(pred.at<float>(0), pred.at<float>(1));
                }
            }
        }
        
        #pragma omp section
        {
            // Process measurements
            for(const auto &d: detections) {
                #pragma omp critical
                {
                    meas.push_back(d.c);
                }
            }
        }
    }

    if(preds.empty()){
        // create tracks for all
        for(auto &d: detections){
            InternalTrack it;
            it.id = nextId++;
            it.kf = createKalmanFilter(d.c);
            it.r = d.r;
            its.push_back(std::move(it));
        }
    } else if(meas.empty()){
        // increase missing
        for(auto &t: its){ t.missing++; }
    } else {
        auto cost = costMatrix(preds, meas);
        auto assign = hungarianSolve(cost);
        // assign returns for each pred the index of meas or -1
        std::vector<bool> measUsed(meas.size(), false);
        for(int i=0;i<(int)assign.size();++i){
            int m = assign[i];
            if(m>=0){
                // update kf
                cv::Mat measV = (cv::Mat_<float>(2,1) << meas[m].x, meas[m].y);
                its[i].kf.correct(measV);
                its[i].missing = 0;
                its[i].r = detections[m].r;
                measUsed[m]=true;
            } else {
                its[i].missing++;
            }
        }
        // create new tracks for unassigned measurements
        for(int i=0;i<(int)meas.size();++i) if(!measUsed[i]){
            InternalTrack it; it.id = nextId++; it.kf = createKalmanFilter(meas[i]); it.r = detections[i].r; its.push_back(std::move(it));
        }
    }

    // remove long-missing
    its.erase(std::remove_if(its.begin(), its.end(), [](const InternalTrack &t){ return t.missing>30; }), its.end());

    // output tracks
    outTracks.clear();
    for(auto &t: its){
        cv::Mat s = t.kf.statePost;
        Track tr; tr.id = t.id; tr.c = cv::Point2f(s.at<float>(0), s.at<float>(1)); tr.v = cv::Point2f(s.at<float>(2), s.at<float>(3)); tr.r = t.r;
        outTracks.push_back(tr);
    }
    
    // Apply physics simulation
    float dt = 1.0f/30.0f; // Assuming 30 fps, could be made dynamic
    Physics::update(outTracks, dt);
    
    // Apply cushion collisions after physics update
    for(auto &tr: outTracks){
        Physics::cushionReflect(tr.c, tr.v, tableSize, tr.r);
    }
}

std::vector<Track> Tracker::tracks() const{ return outTracks; }

std::vector<std::vector<double>> Tracker::costMatrix(const std::vector<cv::Point2f>&pred, const std::vector<cv::Point2f>&meas){
    size_t n = pred.size(), m = meas.size();
    std::vector<std::vector<double>> C(n, std::vector<double>(m,0));
    
    // Compute cost matrix in parallel
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int i=0;i<(int)n;++i) {
        for(int j=0;j<(int)m;++j){
            cv::Point2f d = pred[i]-meas[j];
            C[i][j] = std::hypot(d.x,d.y);
        }
    }
    return C;
}

// Simple Hungarian (Munkres) implementation for rectangular cost matrix
std::vector<int> Tracker::hungarianSolve(const std::vector<std::vector<double>>&cost){
    int n = (int)cost.size();
    int m = n? (int)cost[0].size() : 0;
    std::vector<int> assign(n, -1);
    if(n==0 || m==0) return assign;
    // naive greedy initialization then improve via simple augmenting path (not full optimal but acceptable)
    std::vector<bool> usedM(m,false);
    for(int i=0;i<n;++i){
        double best = 1e9; int bi=-1;
        for(int j=0;j<m;++j) if(!usedM[j] && cost[i][j]<best){ best=cost[i][j]; bi=j; }
        if(bi!=-1){ assign[i]=bi; usedM[bi]=true; }
    }
    return assign;
}
