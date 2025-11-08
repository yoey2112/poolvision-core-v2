#pragma once
#include "../game/MatchSystem.hpp"
#include "../game/GameState.hpp"
#include "../track/Tracker.hpp"
#include "UITheme.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <functional>

namespace pv {

/**
 * @brief Professional match UI with enhanced layout and real-time statistics
 * 
 * Provides a comprehensive match interface with docked panels, live statistics,
 * and professional-grade visualization for competitive play.
 */
class MatchUI {
public:
    /**
     * @brief UI panel configuration
     */
    struct PanelConfig {
        bool enabled;
        cv::Rect rect;
        bool draggable;
        bool resizable;
        int minWidth;
        int minHeight;
        float alpha;        // Transparency level
        
        PanelConfig() : enabled(true), draggable(true), resizable(true),
                       minWidth(200), minHeight(150), alpha(0.9f) {}
    };

    /**
     * @brief Docked panel types
     */
    enum class PanelType {
        BirdsEyeView,      // Top-down table view
        GameStats,         // Live match statistics
        ShotClock,         // Shot timer
        MatchInfo,         // Match configuration and status
        PlayerProfiles,    // Player information and avatars
        ChatLog,           // Commentary/notes
        Controls          // Match control buttons
    };

    /**
     * @brief UI theme configurations
     */
    struct MatchTheme {
        cv::Scalar primaryColor;
        cv::Scalar accentColor;
        cv::Scalar backgroundColor;
        cv::Scalar textColor;
        cv::Scalar panelColor;
        cv::Scalar borderColor;
        float glassOpacity;
        bool animated;
        
        MatchTheme() : primaryColor(UITheme::Colors::TableGreen),
                      accentColor(UITheme::Colors::NeonCyan),
                      backgroundColor(UITheme::Colors::DarkBg),
                      textColor(UITheme::Colors::TextPrimary),
                      panelColor(UITheme::Colors::MediumBg),
                      borderColor(UITheme::Colors::BorderColor),
                      glassOpacity(0.8f), animated(true) {}
    };

    /**
     * @brief Animation states for UI elements
     */
    struct AnimationState {
        float slideProgress;
        float fadeProgress;
        float pulsePhase;
        bool isTransitioning;
        std::chrono::steady_clock::time_point startTime;
        
        AnimationState() : slideProgress(0.0f), fadeProgress(1.0f), pulsePhase(0.0f),
                          isTransitioning(false) {}
    };

    /**
     * @brief Match UI configuration
     */
    struct UIConfig {
        bool fullscreen;
        bool showAllPanels;
        bool autoHide;           // Hide panels when inactive
        double autoHideDelay;    // Seconds before auto-hide
        bool showNotifications;
        bool soundEnabled;
        int uiScale;            // 100 = normal, 150 = large, etc.
        PanelConfig panels[7];   // One for each panel type
        MatchTheme theme;
        
        UIConfig() : fullscreen(false), showAllPanels(true), autoHide(false),
                    autoHideDelay(5.0), showNotifications(true), soundEnabled(true),
                    uiScale(100) {
            // Initialize default panel positions
            initializeDefaultPanels();
        }
        
        void initializeDefaultPanels();
    };

    /**
     * @brief Construct a new Match UI
     */
    MatchUI(std::shared_ptr<MatchSystem> matchSystem, GameState& gameState, Tracker& tracker);
    ~MatchUI() = default;

    /**
     * @brief Render the complete match interface
     */
    void render(cv::Mat& frame);

    /**
     * @brief Handle mouse events
     */
    void handleMouseDown(const cv::Point& pos);
    void handleMouseMove(const cv::Point& pos);
    void handleMouseUp(const cv::Point& pos);
    void handleMouseWheel(int delta, const cv::Point& pos);

    /**
     * @brief Handle keyboard input
     */
    void handleKeyPress(int key);

    /**
     * @brief Update UI state and animations
     */
    void update(double deltaTime);

    /**
     * @brief Set UI configuration
     */
    void setConfig(const UIConfig& config) { config_ = config; }
    
    /**
     * @brief Get current configuration
     */
    const UIConfig& getConfig() const { return config_; }

    /**
     * @brief Enable/disable specific panel
     */
    void setPanelEnabled(PanelType panel, bool enabled);
    
    /**
     * @brief Toggle panel visibility
     */
    void togglePanel(PanelType panel);
    
    /**
     * @brief Reset panels to default positions
     */
    void resetPanelLayout();

    /**
     * @brief Set callbacks for match events
     */
    void setMatchEventCallback(std::function<void(const std::string&)> callback) {
        matchEventCallback_ = callback;
    }

private:
    std::shared_ptr<MatchSystem> matchSystem_;
    GameState& gameState_;
    Tracker& tracker_;
    UIConfig config_;
    AnimationState animations_[7];  // One for each panel
    
    // Panel management
    PanelType activePanelType_;
    bool isDragging_;
    bool isResizing_;
    cv::Point dragOffset_;
    cv::Point lastMousePos_;
    std::chrono::steady_clock::time_point lastActivity_;
    
    // Callbacks
    std::function<void(const std::string&)> matchEventCallback_;

    // Rendering methods
    void renderMainView(cv::Mat& frame);
    void renderBirdsEyePanel(cv::Mat& frame, const cv::Rect& panelRect);
    void renderGameStatsPanel(cv::Mat& frame, const cv::Rect& panelRect);
    void renderShotClockPanel(cv::Mat& frame, const cv::Rect& panelRect);
    void renderMatchInfoPanel(cv::Mat& frame, const cv::Rect& panelRect);
    void renderPlayerProfilesPanel(cv::Mat& frame, const cv::Rect& panelRect);
    void renderChatLogPanel(cv::Mat& frame, const cv::Rect& panelRect);
    void renderControlsPanel(cv::Mat& frame, const cv::Rect& panelRect);

    // Panel utilities
    void renderPanel(cv::Mat& frame, const cv::Rect& rect, PanelType type);
    void renderPanelFrame(cv::Mat& frame, const cv::Rect& rect, const std::string& title, bool isActive = false);
    bool isPointInPanel(const cv::Point& pos, PanelType panel) const;
    cv::Rect getPanelRect(PanelType panel) const;
    cv::Rect getPanelContentRect(const cv::Rect& panelRect) const;

    // Specialized rendering components
    void renderScoreboard(cv::Mat& frame, const cv::Rect& rect);
    void renderPlayerCard(cv::Mat& frame, const cv::Rect& rect, const MatchSystem::MatchPlayer& player, 
                         const MatchSystem::LiveStats& stats, bool isActive);
    void renderShotClockDisplay(cv::Mat& frame, const cv::Rect& rect, const MatchSystem::ShotClock& shotClock);
    void renderMatchProgress(cv::Mat& frame, const cv::Rect& rect);
    void renderLiveStatistics(cv::Mat& frame, const cv::Rect& rect, const MatchSystem::LiveStats& stats);
    void renderTableOverview(cv::Mat& frame, const cv::Rect& rect);
    void renderRecentShots(cv::Mat& frame, const cv::Rect& rect, const std::vector<double>& shotTimes);

    // Chart rendering
    void renderLineChart(cv::Mat& frame, const cv::Rect& rect, const std::vector<double>& data, 
                        const std::string& title, cv::Scalar color = UITheme::Colors::NeonCyan);
    void renderBarChart(cv::Mat& frame, const cv::Rect& rect, const std::vector<std::pair<std::string, double>>& data,
                       const std::string& title);
    void renderProgressRing(cv::Mat& frame, const cv::Point& center, int radius, 
                           double progress, cv::Scalar color);

    // Animation and effects
    void updateAnimations(double deltaTime);
    void animatePanel(PanelType panel, bool fadeIn);
    void renderGlassEffect(cv::Mat& frame, const cv::Rect& rect, float opacity = 0.8f);
    void renderGlow(cv::Mat& frame, const cv::Rect& rect, cv::Scalar color, int intensity = 10);

    // Input handling
    PanelType findPanelAtPosition(const cv::Point& pos) const;
    bool isResizeHandle(const cv::Point& pos, PanelType panel) const;
    void startPanelDrag(PanelType panel, const cv::Point& pos);
    void updatePanelDrag(const cv::Point& pos);
    void endPanelDrag();
    void startPanelResize(PanelType panel, const cv::Point& pos);
    void updatePanelResize(const cv::Point& pos);
    void endPanelResize();

    // Utility methods
    void clampPanelToScreen(PanelType panel);
    cv::Scalar getAccentColor(PanelType panel) const;
    std::string formatTime(double seconds) const;
    std::string formatDuration(double minutes) const;
    std::string formatPercentage(double value) const;

    // Constants
    static constexpr int PANEL_HEADER_HEIGHT = 25;
    static constexpr int PANEL_MARGIN = 10;
    static constexpr int RESIZE_HANDLE_SIZE = 12;
    static constexpr int MIN_PANEL_WIDTH = 200;
    static constexpr int MIN_PANEL_HEIGHT = 150;
};

} // namespace pv