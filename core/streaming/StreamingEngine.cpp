#include "StreamingEngine.hpp"
#include "OverlayManager.hpp"
#include "TemplateSystem.hpp"
#include "OBSInterface.hpp"
#include "PlatformAPIs.hpp"
#include <iostream>
#include <chrono>
#include <chrono>

namespace pv {
namespace streaming {

StreamingEngine::StreamingEngine() {
    stats_.startTime = std::chrono::steady_clock::now();
}

StreamingEngine::~StreamingEngine() {
    shutdown();
}

bool StreamingEngine::initialize() {
    if (isInitialized_) {
        return true;
    }
    
    try {
        // Initialize core components
        overlayManager_ = std::make_unique<OverlayManager>();
        templateSystem_ = std::make_unique<TemplateSystem>();
        obsInterface_ = std::make_unique<OBSInterface>();
        
        // Load default templates
        if (!templateSystem_->loadPresetTemplates()) {
            std::cerr << "Warning: Failed to load preset templates" << std::endl;
        }
        
        isInitialized_ = true;
        logStreamingEvent("StreamingEngine initialized successfully");
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize StreamingEngine: " << e.what() << std::endl;
        return false;
    }
}

void StreamingEngine::shutdown() {
    if (!isInitialized_) {
        return;
    }
    
    if (isStreaming_) {
        stopStreaming();
    }
    
    if (isOBSConnected()) {
        disconnectFromOBS();
    }
    
    // Reset components
    platformAPI_.reset();
    obsInterface_.reset();
    overlayManager_.reset();
    templateSystem_.reset();
    
    isInitialized_ = false;
    logStreamingEvent("StreamingEngine shut down");
}

bool StreamingEngine::connectToOBS() {
    if (!isInitialized_ || !obsInterface_) {
        return false;
    }
    
    try {
        bool connected = obsInterface_->connect();
        if (connected) {
            logStreamingEvent("Connected to OBS Studio");
        }
        return connected;
    } catch (const std::exception& e) {
        std::cerr << "Failed to connect to OBS: " << e.what() << std::endl;
        return false;
    }
}

void StreamingEngine::disconnectFromOBS() {
    if (obsInterface_) {
        obsInterface_->disconnect();
        logStreamingEvent("Disconnected from OBS Studio");
    }
}

bool StreamingEngine::isOBSConnected() const {
    return obsInterface_ && obsInterface_->isConnected();
}

void StreamingEngine::setPlatform(Platform platform) {
    if (currentPlatform_ == platform) {
        return;
    }
    
    currentPlatform_ = platform;
    
    // Create appropriate platform API
    switch (platform) {
        case Platform::Facebook:
            platformAPI_ = std::make_unique<FacebookGamingAPI>();
            break;
        case Platform::YouTube:
            platformAPI_ = std::make_unique<YouTubeGamingAPI>();
            break;
        case Platform::Twitch:
            platformAPI_ = std::make_unique<TwitchAPI>();
            break;
        case Platform::None:
            platformAPI_.reset();
            break;
    }
    
    logStreamingEvent("Platform set to: " + std::to_string(static_cast<int>(platform)));
}

bool StreamingEngine::authenticatePlatform(const std::string& apiKey) {
    if (!platformAPI_) {
        return false;
    }
    
    try {
        bool authenticated = platformAPI_->authenticate(apiKey);
        if (authenticated) {
            logStreamingEvent("Platform authentication successful");
        }
        return authenticated;
    } catch (const std::exception& e) {
        std::cerr << "Platform authentication failed: " << e.what() << std::endl;
        return false;
    }
}

bool StreamingEngine::loadTemplate(const std::string& templateId) {
    if (!templateSystem_) {
        return false;
    }
    
    try {
        auto template_opt = templateSystem_->loadTemplate(templateId);
        if (!template_opt) {
            return false;
        }
        
        currentTemplate_ = *template_opt;
        
        // Apply template to overlay manager
        if (overlayManager_) {
            overlayManager_->loadTemplate(currentTemplate_);
        }
        
        logStreamingEvent("Template loaded: " + templateId);
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load template: " << e.what() << std::endl;
        return false;
    }
}

std::vector<OverlayTemplate> StreamingEngine::getAvailableTemplates() const {
    if (!templateSystem_) {
        return {};
    }
    
    return templateSystem_->getPresetTemplates();
}

bool StreamingEngine::saveCustomTemplate(const OverlayTemplate& overlayTemplate) {
    if (!templateSystem_) {
        return false;
    }
    
    try {
        bool saved = templateSystem_->saveTemplate(overlayTemplate);
        if (saved) {
            logStreamingEvent("Custom template saved: " + overlayTemplate.id);
        }
        return saved;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save custom template: " << e.what() << std::endl;
        return false;
    }
}

void StreamingEngine::updateOverlayData(const OverlayData& data) {
    currentData_ = data;
    
    if (overlayManager_ && isStreaming_) {
        overlayManager_->updateData(data);
        updatePerformanceStats();
    }
}

void StreamingEngine::updateGameState(const GameState& gameState) {
    // Convert GameState to OverlayData
    OverlayData data = currentData_;
    
    // Update game-specific information
    data.currentPlayer = (gameState.getCurrentTurn() == PlayerTurn::Player1) ? "Player 1" : "Player 2";
    data.isBreakShot = gameState.isBreakShot();
    data.matchStatus = gameState.isGameOver() ? "Game Over" : "In Progress";
    
    // Calculate game progress (simplified)
    auto remainingBalls = gameState.getRemainingBalls(BallGroup::Solids);
    remainingBalls.insert(remainingBalls.end(), 
                         gameState.getRemainingBalls(BallGroup::Stripes).begin(),
                         gameState.getRemainingBalls(BallGroup::Stripes).end());
    
    data.gameProgress = 1.0f - (static_cast<float>(remainingBalls.size()) / 15.0f);
    data.ballsRemaining = remainingBalls;
    
    // Get shot suggestions
    auto suggestions = gameState.getSuggestedShots();
    if (!suggestions.empty()) {
        data.suggestedShot = "Target Ball: " + std::to_string(suggestions[0].ballPotted);
    }
    
    updateOverlayData(data);
}

void StreamingEngine::updatePlayerStats(const PlayerInfo& player1, const PlayerInfo& player2) {
    // Copy player data to overlay data structure
    currentData_.player1Name = player1.name;
    currentData_.player1Score = player1.score;
    currentData_.player2Name = player2.name; 
    currentData_.player2Score = player2.score;
    
    if (overlayManager_ && isStreaming_) {
        overlayManager_->updateData(currentData_);
    }
}

bool StreamingEngine::startStreaming(const StreamMetadata& metadata) {
    if (!isInitialized_ || !isOBSConnected()) {
        return false;
    }
    
    try {
        currentMetadata_ = metadata;
        
        // Update platform metadata if connected
        if (platformAPI_) {
            platformAPI_->updateStreamMetadata(metadata);
        }
        
        // Start OBS streaming
        bool started = obsInterface_->startStreaming();
        if (started) {
            isStreaming_ = true;
            stats_.startTime = std::chrono::steady_clock::now();
            logStreamingEvent("Streaming started: " + metadata.title);
        }
        
        return started;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to start streaming: " << e.what() << std::endl;
        return false;
    }
}

void StreamingEngine::stopStreaming() {
    if (!isStreaming_) {
        return;
    }
    
    try {
        if (obsInterface_) {
            obsInterface_->stopStreaming();
        }
        
        if (platformAPI_) {
            StreamMetadata endMetadata = currentMetadata_;
            endMetadata.isLive = false;
            platformAPI_->updateStreamMetadata(endMetadata);
        }
        
        isStreaming_ = false;
        logStreamingEvent("Streaming stopped");
        
    } catch (const std::exception& e) {
        std::cerr << "Error stopping streaming: " << e.what() << std::endl;
    }
}

void StreamingEngine::enableAdvancedEditor(bool enable) {
    advancedEditorEnabled_ = enable;
    
    if (overlayManager_) {
        overlayManager_->setEditorMode(enable);
    }
    
    logStreamingEvent(enable ? "Advanced editor enabled" : "Advanced editor disabled");
}

bool StreamingEngine::addOverlayElement(const OverlayElement& element) {
    if (!overlayManager_ || !advancedEditorEnabled_) {
        return false;
    }
    
    try {
        bool added = overlayManager_->addElement(element);
        if (added) {
            logStreamingEvent("Overlay element added: " + element.id);
        }
        return added;
    } catch (const std::exception& e) {
        std::cerr << "Failed to add overlay element: " << e.what() << std::endl;
        return false;
    }
}

bool StreamingEngine::removeOverlayElement(const std::string& elementId) {
    if (!overlayManager_ || !advancedEditorEnabled_) {
        return false;
    }
    
    try {
        bool removed = overlayManager_->removeElement(elementId);
        if (removed) {
            logStreamingEvent("Overlay element removed: " + elementId);
        }
        return removed;
    } catch (const std::exception& e) {
        std::cerr << "Failed to remove overlay element: " << e.what() << std::endl;
        return false;
    }
}

bool StreamingEngine::moveOverlayElement(const std::string& elementId, cv::Point2f newPosition) {
    if (!overlayManager_ || !advancedEditorEnabled_) {
        return false;
    }
    
    return overlayManager_->moveElement(elementId, newPosition);
}

bool StreamingEngine::resizeOverlayElement(const std::string& elementId, cv::Size2f newSize) {
    if (!overlayManager_ || !advancedEditorEnabled_) {
        return false;
    }
    
    return overlayManager_->resizeElement(elementId, newSize);
}

StreamingEngine::StreamingStats StreamingEngine::getStreamingStats() const {
    updatePerformanceStats();
    return stats_;
}

void StreamingEngine::updatePerformanceStats() const {
    // Update performance statistics
    stats_.framesRendered++;
    
    // Calculate memory usage (simplified)
    stats_.memoryUsage = sizeof(*this);
    if (overlayManager_) {
        stats_.memoryUsage += overlayManager_->getMemoryUsage();
    }
    
    // Calculate average latency (placeholder)
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - stats_.startTime);
    stats_.averageLatency = static_cast<double>(duration.count()) / stats_.framesRendered;
    
    // Check if performance is optimal
    stats_.isOptimal = (stats_.averageLatency < 100.0 && stats_.memoryUsage < 50 * 1024 * 1024);
}

bool StreamingEngine::validatePlatformConnection() {
    return platformAPI_ && platformAPI_->isConnected();
}

void StreamingEngine::logStreamingEvent(const std::string& event) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::cout << "[STREAMING] " << std::ctime(&time_t) << ": " << event << std::endl;
}

// Factory Implementation
std::unique_ptr<StreamingEngine> StreamingEngineFactory::createStandardEngine() {
    auto engine = std::make_unique<StreamingEngine>();
    if (engine->initialize()) {
        return engine;
    }
    return nullptr;
}

std::unique_ptr<StreamingEngine> StreamingEngineFactory::createPerformanceOptimizedEngine() {
    auto engine = std::make_unique<StreamingEngine>();
    if (engine->initialize()) {
        // Apply performance optimizations
        return engine;
    }
    return nullptr;
}

std::unique_ptr<StreamingEngine> StreamingEngineFactory::createDevelopmentEngine() {
    auto engine = std::make_unique<StreamingEngine>();
    if (engine->initialize()) {
        // Apply development-friendly settings
        return engine;
    }
    return nullptr;
}

} // namespace streaming
} // namespace pv