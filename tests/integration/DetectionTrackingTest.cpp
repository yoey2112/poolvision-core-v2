#include <gtest/gtest.h>
#include "../../core/detect/classical/BallDetector.hpp"
#include "../../core/track/Tracker.hpp"
#include "../../core/events/EventEngine.hpp"
#include <opencv2/opencv.hpp>

using namespace pv;

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a sequence of test frames
        for(int i = 0; i < 10; i++) {
            cv::Mat frame = cv::Mat::zeros(500, 1000, CV_8UC3);
            // Moving ball
            cv::Point center(100 + i*20, 250);
            cv::circle(frame, center, 10, cv::Scalar(255,255,255), -1);
            frames.push_back(frame);
        }
        
        // Setup components
        detector.params.minR = 5;
        detector.params.maxR = 15;
        
        tracker.setTableSize({1000, 500});
        
        events.loadTable("test_table.yaml");
    }
    
    std::vector<cv::Mat> frames;
    BallDetector detector;
    Tracker tracker;
    EventEngine events;
};

TEST_F(IntegrationTest, DetectionToTracking) {
    std::vector<Track> tracks;
    double timestamp = 0.0;
    double dt = 1.0/30.0;
    
    for(const auto &frame: frames) {
        // Detection
        auto balls = detector.detect(frame);
        ASSERT_EQ(balls.size(), 1); // Should detect one ball
        
        // Tracking
        tracker.update(balls, timestamp);
        auto current_tracks = tracker.tracks();
        
        if(!current_tracks.empty()) {
            EXPECT_EQ(current_tracks.size(), 1);
            auto &track = current_tracks[0];
            
            // Verify tracking is reasonable
            EXPECT_NEAR(track.c.x, balls[0].c.x, 5.0);
            EXPECT_NEAR(track.c.y, balls[0].c.y, 5.0);
            
            // Velocity should be roughly 20 pixels per frame in x direction
            if(timestamp > dt) {
                EXPECT_NEAR(track.v.x, 20.0f, 5.0f);
                EXPECT_NEAR(track.v.y, 0.0f, 5.0f);
            }
        }
        
        timestamp += dt;
    }
}

TEST_F(IntegrationTest, EventDetection) {
    // Create a sequence showing a ball going into a pocket
    std::vector<cv::Mat> pocketFrames;
    for(int i = 0; i < 10; i++) {
        cv::Mat frame = cv::Mat::zeros(500, 1000, CV_8UC3);
        // Ball moving towards corner pocket
        cv::Point center(80 - i*8, 80 - i*8);
        cv::circle(frame, center, 10, cv::Scalar(255,255,255), -1);
        pocketFrames.push_back(frame);
    }
    
    double timestamp = 0.0;
    double dt = 1.0/30.0;
    bool pocket_detected = false;
    
    for(const auto &frame: pocketFrames) {
        auto balls = detector.detect(frame);
        tracker.update(balls, timestamp);
        auto tracks = tracker.tracks();
        
        auto events_detected = events.detectPocketed(balls, tracks, timestamp);
        if(!events_detected.empty()) {
            pocket_detected = true;
            EXPECT_EQ(events_detected[0].pocket_idx, 0); // Should be top-left pocket
        }
        
        timestamp += dt;
    }
    
    EXPECT_TRUE(pocket_detected);
}