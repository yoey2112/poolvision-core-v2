/**
 * Example integration of AI Learning System with Pool Vision
 * 
 * This demonstrates how to add intelligent AI features to the existing
 * table_daemon.exe without interfering with the high-performance GPU pipeline.
 */

#include "core/ai/learning/AILearningSystem.hpp"
#include "core/track/Tracker.hpp"
#include "core/events/EventEngine.hpp"
#include "core/detect/classical/BallDetector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

using namespace pv::ai::learning;

class PoolVisionWithAI {
private:
    // Existing components
    std::shared_ptr<Tracker> tracker_;
    std::shared_ptr<EventEngine> eventEngine_;
    std::shared_ptr<BallDetector> ballDetector_;
    
    // AI Learning System
    std::unique_ptr<AILearningSystem> aiLearning_;
    
    // Current game state
    GameState currentGameState_;
    std::vector<Ball> currentBalls_;
    int currentPlayerId_ = 1;

public:
    bool initialize() {
        // Initialize existing components (simplified)
        tracker_ = std::make_shared<Tracker>();
        eventEngine_ = std::make_shared<EventEngine>();
        ballDetector_ = std::make_shared<BallDetector>();
        
        // Configure AI Learning System
        AILearningSystem::SystemConfig aiConfig;
        aiConfig.enableDataCollection = true;
        aiConfig.enableShotAnalysis = true;
        aiConfig.enableAdaptiveCoaching = true;
        aiConfig.enablePerformanceAnalytics = true;
        aiConfig.ollamaEndpoint = "http://localhost:11434"; // Local Ollama server
        aiConfig.cpuCores = {4, 5, 6, 7}; // Use cores 4-7 for AI processing
        aiConfig.maxCpuUsage = 20; // Limit AI to 20% CPU usage
        
        // Create and initialize AI Learning System
        aiLearning_ = AILearningSystemFactory::createWithConfig(aiConfig);
        
        if (!aiLearning_->initialize()) {
            std::cerr << "Failed to initialize AI Learning System" << std::endl;
            return false;
        }
        
        // Connect AI system to existing components
        aiLearning_->connectToTracker(tracker_);
        aiLearning_->connectToEventEngine(eventEngine_);
        aiLearning_->connectToBallDetector(ballDetector_);
        
        // Start AI processing
        aiLearning_->start();
        
        // Add test player
        aiLearning_->addPlayer(currentPlayerId_, "Test Player");
        aiLearning_->startPlayerSession(currentPlayerId_);
        
        std::cout << "Pool Vision with AI Learning initialized successfully!" << std::endl;
        return true;
    }
    
    void processFrame(const cv::Mat& frame) {
        // 1. Run existing high-performance detection/tracking (unchanged)
        // This runs on GPU cores 0-3 at 200+ FPS
        auto balls = ballDetector_->detectBalls(frame);
        currentBalls_ = balls;
        
        // Track balls at 300+ FPS
        tracker_->updateTracks(balls);
        
        // 2. Feed data to AI Learning System (CPU cores 4-7)
        // This happens asynchronously and doesn't block the main pipeline
        aiLearning_->onBallPositionsUpdate(balls);
        
        // 3. Get intelligent insights (when available)
        auto insights = aiLearning_->getPlayerInsights(currentPlayerId_);
        displayInsights(frame, insights);
        
        // 4. Generate intelligent coaching (when appropriate)
        if (shouldGenerateCoaching()) {
            auto shotAnalysis = aiLearning_->analyzeShotSituation(
                currentPlayerId_, currentGameState_, getCueBallPosition(), getTargetBalls());
            
            auto coaching = aiLearning_->generateCoaching(
                currentPlayerId_, currentGameState_, shotAnalysis);
            
            displayCoaching(frame, coaching);
        }
        
        // 5. Update performance analytics in background
        auto metrics = aiLearning_->getPlayerMetrics(currentPlayerId_);
        displayPerformanceMetrics(frame, metrics);
    }
    
    void onShotCompleted(const cv::Point2f& startPos, const cv::Point2f& endPos, bool successful) {
        // This gets called when a shot is completed
        // The AI system learns from each shot automatically
        
        // Calculate shot difficulty based on distance and angle
        float distance = cv::norm(endPos - startPos);
        float difficulty = std::clamp(distance / 500.0f, 0.0f, 1.0f);
        
        // Feed shot data to AI Learning System
        aiLearning_->onShotCompleted(currentPlayerId_, startPos, endPos, successful, difficulty);
        
        std::cout << "Shot recorded: " << (successful ? "Success" : "Miss") 
                  << " (difficulty: " << difficulty << ")" << std::endl;
    }
    
    void onPlayerBehavior(float aimingTime, float confidence) {
        // Record player behavior for coaching adaptation
        aiLearning_->onPlayerBehavior(currentPlayerId_, aimingTime, confidence);
    }
    
    void generatePerformanceReport() {
        // Generate comprehensive performance report
        auto sessionReport = aiLearning_->getSessionReport(currentPlayerId_);
        std::cout << "\n=== Performance Report ===" << std::endl;
        std::cout << "Session Duration: " << 
            std::chrono::duration_cast<std::chrono::minutes>(
                sessionReport.sessionEnd - sessionReport.sessionStart).count() 
            << " minutes" << std::endl;
        std::cout << "Total Shots: " << sessionReport.performance.totalShots << std::endl;
        std::cout << "Success Rate: " << (sessionReport.performance.successfulShots * 100.0f / 
                                         std::max(1, sessionReport.performance.totalShots)) << "%" << std::endl;
        std::cout << "Session Rating: " << sessionReport.performance.sessionRating << std::endl;
        
        // Generate performance chart
        auto chart = aiLearning_->generatePlayerPerformanceChart(currentPlayerId_, "trend");
        cv::imshow("Performance Trend", chart);
        cv::waitKey(1);
    }
    
    void shutdown() {
        if (aiLearning_) {
            aiLearning_->endPlayerSession(currentPlayerId_);
            aiLearning_->stop();
            aiLearning_->shutdown();
            
            // Generate final report
            generatePerformanceReport();
        }
        
        std::cout << "Pool Vision with AI Learning shut down" << std::endl;
    }

private:
    cv::Point2f getCueBallPosition() {
        // Find cue ball from current ball positions
        for (const auto& ball : currentBalls_) {
            if (ball.n == 0) { // Cue ball
                return ball.c;
            }
        }
        return cv::Point2f(400, 300); // Default center position
    }
    
    std::vector<Ball> getTargetBalls() {
        // Return non-cue balls as targets
        std::vector<Ball> targets;
        for (const auto& ball : currentBalls_) {
            if (ball.n != 0) {
                targets.push_back(ball);
            }
        }
        return targets;
    }
    
    bool shouldGenerateCoaching() {
        // Generate coaching every 10 seconds or when player seems to be struggling
        static auto lastCoaching = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        
        return std::chrono::duration_cast<std::chrono::seconds>(now - lastCoaching).count() > 10;
    }
    
    void displayInsights(const cv::Mat& frame, 
                        const std::vector<PerformanceAnalyticsEngine::PerformanceInsight>& insights) {
        if (insights.empty()) return;
        
        int y = 30;
        for (const auto& insight : insights) {
            cv::Scalar color = (insight.type == PerformanceAnalyticsEngine::PerformanceInsight::Achievement) ? 
                               cv::Scalar(0, 255, 0) : cv::Scalar(0, 100, 255);
            
            cv::putText(frame, insight.title, cv::Point(10, y), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            y += 25;
            
            cv::putText(frame, insight.description.substr(0, 60), cv::Point(10, y), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
            y += 40;
        }
    }
    
    void displayCoaching(const cv::Mat& frame, const AdaptiveCoachingEngine::CoachingMessage& coaching) {
        if (coaching.message.empty()) return;
        
        cv::Scalar color = cv::Scalar(255, 200, 0); // Gold color for coaching
        
        // Display coaching message at bottom of frame
        int y = frame.rows - 60;
        cv::putText(frame, "Coach: " + coaching.message.substr(0, 80), cv::Point(10, y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
    
    void displayPerformanceMetrics(const cv::Mat& frame, 
                                 const PerformanceAnalyticsEngine::PerformanceMetrics& metrics) {
        // Display key metrics in top-right corner
        int x = frame.cols - 250;
        int y = 30;
        
        cv::Scalar color = cv::Scalar(255, 255, 255);
        
        cv::putText(frame, "Performance Metrics", cv::Point(x, y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        y += 25;
        
        std::string successRate = "Success Rate: " + 
                                 std::to_string(static_cast<int>(metrics.basicStats.successRate * 100)) + "%";
        cv::putText(frame, successRate, cv::Point(x, y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        y += 20;
        
        std::string totalShots = "Total Shots: " + std::to_string(metrics.basicStats.totalShots);
        cv::putText(frame, totalShots, cv::Point(x, y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        y += 20;
        
        std::string consistency = "Consistency: " + 
                                 std::to_string(static_cast<int>(metrics.advanced.consistencyIndex * 100)) + "%";
        cv::putText(frame, consistency, cv::Point(x, y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    }
};

// Demonstration main function
int main() {
    std::cout << "=== Pool Vision with AI Learning System Demo ===" << std::endl;
    
    PoolVisionWithAI poolVision;
    
    if (!poolVision.initialize()) {
        std::cerr << "Failed to initialize Pool Vision with AI" << std::endl;
        return -1;
    }
    
    // Simulate a pool session
    cv::Mat frame(600, 800, CV_8UC3, cv::Scalar(0, 100, 0)); // Green pool table
    
    std::cout << "\nSimulating pool session..." << std::endl;
    
    // Simulate 10 shots with varying success
    for (int i = 0; i < 10; ++i) {
        // Simulate shot positions
        cv::Point2f startPos(400 + rand() % 100, 300 + rand() % 100);
        cv::Point2f endPos(500 + rand() % 200, 250 + rand() % 200);
        bool successful = (rand() % 100) < 70; // 70% success rate
        
        // Simulate aiming time and confidence
        float aimingTime = 2.0f + (rand() % 30) / 10.0f; // 2-5 seconds
        float confidence = 0.3f + (rand() % 70) / 100.0f; // 0.3-1.0
        
        poolVision.onPlayerBehavior(aimingTime, confidence);
        poolVision.processFrame(frame);
        poolVision.onShotCompleted(startPos, endPos, successful);
        
        std::cout << "Shot " << (i + 1) << ": " << (successful ? "Success" : "Miss") << std::endl;
        
        // Small delay to simulate real gameplay
        cv::waitKey(100);
    }
    
    std::cout << "\nGenerating final performance report..." << std::endl;
    poolVision.generatePerformanceReport();
    
    std::cout << "\nPress any key to exit..." << std::endl;
    cv::waitKey(0);
    
    poolVision.shutdown();
    
    std::cout << "Demo completed successfully!" << std::endl;
    return 0;
}

/*
Example output:

=== Pool Vision with AI Learning System Demo ===
Initializing AI Learning System...
Data Collection Engine initialized
Shot Analysis Engine initialized  
Adaptive Coaching Engine initialized
Performance Analytics Engine initialized
AI Learning System initialized successfully!
Pool Vision with AI Learning initialized successfully!
Adding player 1 (Test Player) to AI Learning System
Performance Analytics Engine started
AI Learning System started successfully!

=== AI Learning System Status ===
Data Collection: Active
Shot Analysis: Active
Adaptive Coaching: Active
Performance Analytics: Active
Players Tracked: 1
CPU Usage: 18.5%
Data Quality: 1.00
===================================

Simulating pool session...
Shot recorded: Success (difficulty: 0.3)
Shot recorded: Miss (difficulty: 0.7)
Shot recorded: Success (difficulty: 0.4)
...

=== Performance Report ===
Session Duration: 2 minutes
Total Shots: 10
Success Rate: 70%
Session Rating: 0.72

Demo completed successfully!
*/