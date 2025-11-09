#pragma once

#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <memory>

#include "../../track/modern/ByteTrackMOT.hpp"
#include "../../game/modern/ShotSegmentation.hpp"
#include "../../game/GameState.hpp"
#include "../../util/Types.hpp"

#ifdef USE_OLLAMA
#include "../../ai/CoachingEngine.hpp"
#endif

namespace pv {
namespace modern {

/**
 * SeparatedUIRenderer - Agent Group 5: UI & Integration
 * 
 * Separated UI rendering pipeline that runs at 60 FPS independently
 * of the main inference pipeline (which may run at 200+ FPS).
 * 
 * Features:
 * - Dedicated UI thread with CPU core affinity
 * - Lock-free frame queue management
 * - Overlay rendering (ball detection, tracking, game state)
 * - Birds-eye view tactical rendering
 * - Performance monitoring and metrics
 * - Multiple output formats for streaming/display
 */
class SeparatedUIRenderer {
public:
    struct UIRenderConfig {
        int targetFps = 60;                          // UI refresh rate
        bool enableVSync = true;                     // Prevent screen tearing
        int renderThreadCpuCore = 3;                 // Dedicated CPU core
        bool enableBirdsEyeView = true;              // Top-down tactical view
        bool enableOverlays = true;                  // Ball tracking overlays
        cv::Size overlayResolution{1920, 1080};     // Output resolution
        bool enablePerformanceHUD = true;           // Show performance metrics
        bool enableCoachingOverlay = true;          // Show AI coaching text
        int maxFrameQueueSize = 3;                  // Prevent memory buildup
    };
    
    struct FrameData {
        cv::Mat originalFrame;                       // Original camera frame
        std::vector<Ball> detections;                // Ball detections
        std::vector<pv::modern::ByteTrackMOT::Track> tracks; // Tracking results
        pv::modern::ShotSegmentation::ShotEvent currentShot; // Shot analysis
        GameState gameState;                         // Current game state
        uint64_t frameTimestamp;                     // Frame timestamp
        float inferenceTime;                         // GPU inference time
        float trackingTime;                          // CPU tracking time
        bool hasValidData;                           // Data validity flag
        
#ifdef USE_OLLAMA
        std::string latestCoachingAdvice;            // Latest AI coaching
        bool hasNewCoaching;                         // New coaching available
#endif
        
        FrameData() : hasValidData(false) 
#ifdef USE_OLLAMA
        , hasNewCoaching(false) 
#endif
        {}
    };

private:
    UIRenderConfig config_;
    
    // UI thread management
    std::thread uiRenderThread_;
    std::atomic<bool> renderingActive_{false};
    
    // Frame data pipeline with thread safety
    std::queue<FrameData> frameQueue_;
    std::mutex frameQueueMutex_;
    std::condition_variable frameCondition_;
    
    // Latest rendered frames for external access
    cv::Mat latestCompositeFrame_;
    cv::Mat latestBirdsEyeFrame_;
    cv::Mat latestOverlayFrame_;
    std::mutex outputFramesMutex_;
    
    // Overlay rendering subsystem
    class OverlayRenderer {
    private:
        cv::Mat overlayBuffer_;
        std::vector<cv::Scalar> ballColors_;
        cv::Scalar trackColor_{0, 255, 0};
        cv::Scalar predictionColor_{255, 255, 0};
        cv::Scalar coachingColor_{255, 255, 255};
        cv::Font font_{cv::FONT_HERSHEY_SIMPLEX};
        
    public:
        OverlayRenderer();
        
        void renderBallDetections(cv::Mat& output, const std::vector<Ball>& detections);
        void renderTrackingOverlays(cv::Mat& output, const std::vector<pv::modern::ByteTrackMOT::Track>& tracks);
        void renderShotAnalysis(cv::Mat& output, const pv::modern::ShotSegmentation::ShotEvent& shot);
        void renderGameHUD(cv::Mat& output, const GameState& state);
        void renderPerformanceMetrics(cv::Mat& output, float inferenceTime, float trackingTime, float renderTime);
        
#ifdef USE_OLLAMA
        void renderCoachingOverlay(cv::Mat& output, const std::string& advice);
#endif
        
    private:
        void drawBall(cv::Mat& output, const Ball& ball, const cv::Scalar& color);
        void drawTrack(cv::Mat& output, const pv::modern::ByteTrackMOT::Track& track);
        void drawText(cv::Mat& output, const std::string& text, cv::Point position, 
                     const cv::Scalar& color, float scale = 0.7f);
    };
    
    std::unique_ptr<OverlayRenderer> overlayRenderer_;
    
    // Birds-eye view renderer
    class BirdsEyeRenderer {
    private:
        cv::Mat tableBackground_;
        cv::Mat birdsEyeBuffer_;
        cv::Size tableSize_{800, 400};              // Scaled table dimensions
        cv::Mat homographyMatrix_;                  // Camera view to table view
        std::vector<cv::Point2f> pocketLocations_;
        bool initialized_{false};
        
    public:
        BirdsEyeRenderer();
        
        void initializeTableView(const cv::Size& tableSize);
        void renderTableLayout(cv::Mat& output);
        void renderBallPositions(cv::Mat& output, const std::vector<pv::modern::ByteTrackMOT::Track>& tracks);
        void renderShotPrediction(cv::Mat& output, const pv::modern::ByteTrackMOT::Track& cueBall, 
                                 const cv::Point2f& targetPoint);
        void renderPocketProbabilities(cv::Mat& output, const std::vector<float>& probabilities);
        
        bool isInitialized() const { return initialized_; }
        
    private:
        cv::Point2f transformToTableCoords(const cv::Point2f& cameraPoint);
        void drawPockets(cv::Mat& output);
        void drawTable(cv::Mat& output);
    };
    
    std::unique_ptr<BirdsEyeRenderer> birdsEyeRenderer_;
    
    // Performance monitoring
    struct UIPerformanceMetrics {
        float averageRenderTime = 0.0f;
        float currentFPS = 0.0f;
        float cpuUsage = 0.0f;
        int droppedFrames = 0;
        int totalFramesRendered = 0;
        std::chrono::steady_clock::time_point lastFrameTime;
        std::chrono::steady_clock::time_point startTime;
        
        UIPerformanceMetrics() {
            lastFrameTime = std::chrono::steady_clock::now();
            startTime = lastFrameTime;
        }
    };
    
    UIPerformanceMetrics performanceMetrics_;
    std::mutex metricsMutex_;

public:
    SeparatedUIRenderer(const UIRenderConfig& config = UIRenderConfig{});
    ~SeparatedUIRenderer();
    
    // Main UI interface
    void submitFrameData(const FrameData& frameData);
    cv::Mat getCurrentCompositeFrame();
    cv::Mat getCurrentBirdsEyeView();
    cv::Mat getCurrentOverlayFrame();
    
    // UI system management  
    bool startUIRendering();
    void stopUIRendering();
    bool isRenderingActive() const { return renderingActive_.load(); }
    
    // Configuration management
    void setRenderConfig(const UIRenderConfig& newConfig);
    void enableOverlay(const std::string& overlayType, bool enable);
    void setTableGeometry(const cv::Size& tableSize);
    
    // Performance monitoring
    UIPerformanceMetrics getPerformanceMetrics();
    float getRenderingFPS();
    int getQueuedFrames();
    int getDroppedFrames();
    
    // Export/streaming integration
    cv::Mat getCompositeFrame();                    // Overlay + original
    cv::Mat getBirdsEyeFrame();                     // Table view only
    cv::Mat getSideBySideFrame();                   // Original + birds eye
    
#ifdef USE_OLLAMA
    // AI coaching integration
    void updateCoachingAdvice(const std::string& advice);
#endif

private:
    // UI thread main loop
    void renderingLoop();
    void processFrameData(const FrameData& frameData);
    void compositeFrame(const FrameData& frameData, cv::Mat& output);
    void updatePerformanceMetrics(float renderTime);
    void dropOldFrames();                           // Prevent queue buildup
    void setCpuAffinity();                          // Pin to specific CPU core
    
    // Frame processing
    void renderMainView(const FrameData& frameData, cv::Mat& output);
    void renderBirdsEyeView(const FrameData& frameData, cv::Mat& output);
    void renderCompositeView(const FrameData& frameData, cv::Mat& output);
    
    // Utility functions
    bool shouldDropFrame(const FrameData& frameData);
    void logPerformanceMetrics();
};

/**
 * UIRendererFactory - Helper factory for creating UI renderers
 */
class UIRendererFactory {
public:
    static std::unique_ptr<SeparatedUIRenderer> createDefault();
    static std::unique_ptr<SeparatedUIRenderer> createHighPerformance();
    static std::unique_ptr<SeparatedUIRenderer> createLowLatency();
    static std::unique_ptr<SeparatedUIRenderer> createStreamingOptimized();
    
private:
    static SeparatedUIRenderer::UIRenderConfig getDefaultConfig();
    static SeparatedUIRenderer::UIRenderConfig getHighPerformanceConfig();
    static SeparatedUIRenderer::UIRenderConfig getLowLatencyConfig();
    static SeparatedUIRenderer::UIRenderConfig getStreamingConfig();
};

} // namespace modern
} // namespace pv