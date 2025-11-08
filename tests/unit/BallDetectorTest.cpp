#include <gtest/gtest.h>
#include "../../core/detect/classical/BallDetector.hpp"
#include "../../core/util/ColorLab.hpp"
#include <opencv2/opencv.hpp>

using namespace pv;

class BallDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test image with a ball
        img = cv::Mat::zeros(200, 200, CV_8UC3);
        cv::circle(img, cv::Point(100, 100), 10, cv::Scalar(255,255,255), -1);
        
        // Load test color config
        std::stringstream ss;
        ss << "cue: [70, -5, 20]\n";
        ss << "1: [50, 10, 20]\n";
        detector.loadColors("test_colors.yaml");
    }
    
    cv::Mat img;
    BallDetector detector;
};

TEST_F(BallDetectorTest, DetectSingleBall) {
    auto balls = detector.detect(img);
    ASSERT_EQ(balls.size(), 1);
    
    auto &ball = balls[0];
    EXPECT_NEAR(ball.c.x, 100, 2);
    EXPECT_NEAR(ball.c.y, 100, 2);
    EXPECT_NEAR(ball.r, 10, 2);
}

TEST_F(BallDetectorTest, DetectMultipleBalls) {
    // Create image with two balls
    cv::Mat img2 = cv::Mat::zeros(200, 200, CV_8UC3);
    cv::circle(img2, cv::Point(50, 50), 10, cv::Scalar(255,255,255), -1);
    cv::circle(img2, cv::Point(150, 150), 10, cv::Scalar(255,255,255), -1);
    
    auto balls = detector.detect(img2);
    ASSERT_EQ(balls.size(), 2);
}

TEST_F(BallDetectorTest, ColorClassification) {
    // Create image with white ball
    cv::Mat imgWhite = cv::Mat::zeros(200, 200, CV_8UC3);
    cv::circle(imgWhite, cv::Point(100, 100), 10, cv::Scalar(255,255,255), -1);
    
    auto balls = detector.detect(imgWhite);
    ASSERT_EQ(balls.size(), 1);
    
    // Should be classified as cue ball (assuming white is closest to cue ball in LAB space)
    EXPECT_EQ(balls[0].label, -1); // -1 is cue ball
}

TEST_F(BallDetectorTest, DetectorParams) {
    // Test parameter effects
    BallDetector d2;
    d2.params.minR = 20; // Bigger than our test ball
    
    auto balls = d2.detect(img);
    EXPECT_EQ(balls.size(), 0); // Should not detect any balls
    
    d2.params.minR = 5;
    balls = d2.detect(img);
    EXPECT_GT(balls.size(), 0); // Should now detect the ball
}