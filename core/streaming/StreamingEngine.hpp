#pragma once
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "StreamingTypes.hpp"
#include "../game/GameState.hpp"
#include "../util/Types.hpp"

namespace pv {
namespace streaming {

// Forward declarations
class OBSInterface;
class PlatformAPI;
class OverlayManager;
class TemplateSystem;

/**
 * Streaming platform types supported
 */
enum class Platform {
    Facebook,
    YouTube,
    Twitch,
    None
};
/**
 * Stream metadata for platform APIs
 */
struct StreamMetadata {
    std::string title;
    std::string game = "Pool/Billiards";
    std::string description;
    std::vector<std::string> tags;
    std::string thumbnailPath;
    bool isLive = false;
};

/**
 * Player information for overlay display
 */
struct PlayerInfo {
    std::string name;
    int score = 0;
    float winRate = 0.0f;
    int gamesPlayed = 0;
    std::string currentStreak;
    cv::Scalar color = cv::Scalar(255, 255, 255);
};

/**
 * Main streaming engine - orchestrates all streaming functionality
 */
class StreamingEngine {
public:
    StreamingEngine();
    ~StreamingEngine();
    
    // Core lifecycle
    bool initialize();
    void shutdown();
    
    // OBS integration
    bool connectToOBS();
    void disconnectFromOBS();
    bool isOBSConnected() const;
    
    // Platform management
    void setPlatform(Platform platform);
    Platform getCurrentPlatform() const { return currentPlatform_; }
    bool authenticatePlatform(const std::string& apiKey);
    
    // Template management
    bool loadTemplate(const std::string& templateId);
    std::vector<OverlayTemplate> getAvailableTemplates() const;
    bool saveCustomTemplate(const OverlayTemplate& overlayTemplate);
    
    // Real-time data updates
    void updateOverlayData(const OverlayData& data);
    void updateGameState(const GameState& gameState);
    void updatePlayerStats(const PlayerInfo& player1, const PlayerInfo& player2);
    
    // Streaming controls
    bool startStreaming(const StreamMetadata& metadata);
    void stopStreaming();
    bool isStreaming() const { return isStreaming_; }
    
    // Advanced editor
    void enableAdvancedEditor(bool enable);
    bool addOverlayElement(const OverlayElement& element);
    bool removeOverlayElement(const std::string& elementId);
    bool moveOverlayElement(const std::string& elementId, cv::Point2f newPosition);
    bool resizeOverlayElement(const std::string& elementId, cv::Size2f newSize);
    
    // Statistics and monitoring
    struct StreamingStats {
        std::chrono::steady_clock::time_point startTime;
        int framesRendered = 0;
        double averageLatency = 0.0;
        size_t memoryUsage = 0;
        bool isOptimal = true;
    };
    
    StreamingStats getStreamingStats() const;

private:
    // Core components
    std::unique_ptr<OBSInterface> obsInterface_;
    std::unique_ptr<PlatformAPI> platformAPI_;
    std::unique_ptr<OverlayManager> overlayManager_;
    std::unique_ptr<TemplateSystem> templateSystem_;
    
    // State management
    Platform currentPlatform_ = Platform::None;
    bool isInitialized_ = false;
    bool isStreaming_ = false;
    bool advancedEditorEnabled_ = false;
    
    // Current data
    OverlayData currentData_;
    OverlayTemplate currentTemplate_;
    StreamMetadata currentMetadata_;
    
    // Performance monitoring
    mutable StreamingStats stats_;
    
    // Helper methods
    void updatePerformanceStats() const;
    bool validatePlatformConnection();
    void logStreamingEvent(const std::string& event);
};

/**
 * Factory for creating streaming engines with different configurations
 */
class StreamingEngineFactory {
public:
    static std::unique_ptr<StreamingEngine> createStandardEngine();
    static std::unique_ptr<StreamingEngine> createPerformanceOptimizedEngine();
    static std::unique_ptr<StreamingEngine> createDevelopmentEngine();
};

} // namespace streaming
} // namespace pv