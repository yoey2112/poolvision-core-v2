#pragma once

#include "modern/ShotSegmentation.hpp"
#include "GameState.hpp"
#include "../track/modern/ByteTrackMOT.hpp"

namespace pv {

/**
 * Integration adapter that connects modern game logic components
 * with the existing Pool Vision system
 */
class ModernGameLogicAdapter {
private:
    std::unique_ptr<modern::GameLogicManager> gameLogicManager_;
    GameState* legacyGameState_;
    bool enabled_;

public:
    explicit ModernGameLogicAdapter(GameState* legacyState, bool enabled = true)
        : legacyGameState_(legacyState), enabled_(enabled) {
        
        if (enabled_) {
            modern::GameLogicManager::Config config;
            config.gameType = modern::PoolRulesEngine::GameType::EightBall;
            config.enableAdvancedPhysics = true;
            config.enableRealTimeValidation = true;
            
            gameLogicManager_ = std::make_unique<modern::GameLogicManager>(
                legacyGameState_, config);
        }
    }
    
    ~ModernGameLogicAdapter() = default;
    
    // Main processing interface
    void processTracks(const std::vector<Track>& tracks, double timestamp) {
        if (enabled_ && gameLogicManager_) {
            gameLogicManager_->processTracks(tracks, timestamp);
        }
    }
    
    // Enhanced game state queries
    bool isShotInProgress() const {
        if (enabled_ && gameLogicManager_) {
            return gameLogicManager_->isShotInProgress();
        }
        return false;
    }
    
    std::string getAdvancedGameState() const {
        if (enabled_ && gameLogicManager_) {
            return gameLogicManager_->getAdvancedGameState();
        }
        return "Advanced game logic disabled";
    }
    
    std::vector<int> getLegalTargets() const {
        if (enabled_ && gameLogicManager_) {
            return gameLogicManager_->getLegalTargets();
        }
        return {};
    }
    
    // Performance monitoring
    uint64_t getShotsProcessed() const {
        if (enabled_ && gameLogicManager_) {
            return gameLogicManager_->getShotsProcessed();
        }
        return 0;
    }
    
    double getAvgValidationTime() const {
        if (enabled_ && gameLogicManager_) {
            return gameLogicManager_->getAvgValidationTime();
        }
        return 0.0;
    }
    
    // Configuration
    bool isEnabled() const { return enabled_; }
    void setEnabled(bool enabled) { enabled_ = enabled; }
};

} // namespace pv