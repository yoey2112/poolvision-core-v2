#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <memory>
#include <chrono>
#include "StreamingTypes.hpp"

namespace pv {
namespace streaming {

/**
 * Manages overlay elements and their rendering for streaming
 */
class OverlayManager {
public:
    OverlayManager();
    ~OverlayManager();
    
    // Template management
    bool loadTemplate(const OverlayTemplate& overlayTemplate);
    OverlayTemplate getCurrentTemplate() const { return currentTemplate_; }
    
    // Data updates
    void updateData(const OverlayData& data);
    void updateElement(const std::string& elementId, const std::map<std::string, std::string>& properties);
    
    // Element management
    bool addElement(const OverlayElement& element);
    bool removeElement(const std::string& elementId);
    bool moveElement(const std::string& elementId, cv::Point2f newPosition);
    bool resizeElement(const std::string& elementId, cv::Size2f newSize);
    OverlayElement* getElement(const std::string& elementId);
    
    // Rendering
    cv::Mat renderOverlay(cv::Size outputSize);
    void setEditorMode(bool enabled) { editorMode_ = enabled; }
    bool isEditorMode() const { return editorMode_; }
    
    // Performance
    size_t getMemoryUsage() const;
    void optimizeForPerformance();
    
    // Animation support
    void setElementAnimation(const std::string& elementId, const std::string& animationType, float duration);
    void updateAnimations(float deltaTime);

private:
    // Core state
    OverlayTemplate currentTemplate_;
    std::map<std::string, OverlayElement> elements_;
    OverlayData currentData_;
    bool editorMode_ = false;
    
    // Rendering resources
    cv::Mat overlayBuffer_;
    std::map<std::string, cv::Mat> elementTextures_;
    
    // Animation system
    struct ElementAnimation {
        std::string type;
        float duration;
        float elapsed;
        cv::Point2f startPos;
        cv::Point2f targetPos;
        cv::Size2f startSize;
        cv::Size2f targetSize;
        float startOpacity;
        float targetOpacity;
    };
    std::map<std::string, ElementAnimation> animations_;
    
    // Rendering methods
    void renderTextElement(cv::Mat& output, const OverlayElement& element);
    void renderImageElement(cv::Mat& output, const OverlayElement& element);
    void renderChartElement(cv::Mat& output, const OverlayElement& element);
    void renderProgressElement(cv::Mat& output, const OverlayElement& element);
    void renderPlayerInfoElement(cv::Mat& output, const OverlayElement& element);
    void renderGameStatsElement(cv::Mat& output, const OverlayElement& element);
    void renderTimerElement(cv::Mat& output, const OverlayElement& element);
    void renderScoreElement(cv::Mat& output, const OverlayElement& element);
    
    // Helper methods
    void applyGlassMorphismEffect(cv::Mat& region, const TemplateStyle& style);
    cv::Scalar interpolateColor(cv::Scalar color1, cv::Scalar color2, float t);
    std::string formatTime(std::chrono::steady_clock::time_point startTime);
    void drawTextWithOutline(cv::Mat& img, const std::string& text, cv::Point2f position, 
                           const TemplateStyle& style, float scale = 1.0f);
    cv::Rect2f getElementBounds(const OverlayElement& element);
    
    // Editor helpers
    void drawElementBorders(cv::Mat& output);
    void drawResizeHandles(cv::Mat& output, const OverlayElement& element);
    
    // Performance tracking
    mutable size_t lastMemoryUsage_ = 0;
    std::chrono::steady_clock::time_point lastOptimization_;
};

} // namespace streaming
} // namespace pv