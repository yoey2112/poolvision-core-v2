#include "OverlayManager.hpp"
#include <chrono>

namespace pv {
namespace streaming {

OverlayManager::OverlayManager() {
    // Initialize overlay buffer
    overlayBuffer_ = cv::Mat::zeros(1920, 1080, CV_8UC4); // Default to 1080p with alpha
    lastOptimization_ = std::chrono::steady_clock::now();
}

OverlayManager::~OverlayManager() = default;

bool OverlayManager::loadTemplate(const OverlayTemplate& overlayTemplate) {
    currentTemplate_ = overlayTemplate;
    elements_.clear();
    
    // Load template elements
    for (const auto& element : overlayTemplate.elements) {
        elements_[element.id] = element;
    }
    
    return true;
}

void OverlayManager::updateData(const OverlayData& data) {
    currentData_ = data;
    
    // Update animations
    auto now = std::chrono::steady_clock::now();
    static auto lastUpdate = now;
    auto deltaTime = std::chrono::duration<float>(now - lastUpdate).count();
    updateAnimations(deltaTime);
    lastUpdate = now;
}

void OverlayManager::updateElement(const std::string& elementId, const std::map<std::string, std::string>& properties) {
    auto it = elements_.find(elementId);
    if (it != elements_.end()) {
        for (const auto& prop : properties) {
            it->second.properties[prop.first] = prop.second;
        }
    }
}

bool OverlayManager::addElement(const OverlayElement& element) {
    if (elements_.find(element.id) != elements_.end()) {
        return false; // Element already exists
    }
    
    elements_[element.id] = element;
    return true;
}

bool OverlayManager::removeElement(const std::string& elementId) {
    auto it = elements_.find(elementId);
    if (it != elements_.end()) {
        elements_.erase(it);
        animations_.erase(elementId);
        return true;
    }
    return false;
}

bool OverlayManager::moveElement(const std::string& elementId, cv::Point2f newPosition) {
    auto it = elements_.find(elementId);
    if (it != elements_.end()) {
        it->second.position = newPosition;
        return true;
    }
    return false;
}

bool OverlayManager::resizeElement(const std::string& elementId, cv::Size2f newSize) {
    auto it = elements_.find(elementId);
    if (it != elements_.end()) {
        it->second.size = newSize;
        return true;
    }
    return false;
}

OverlayElement* OverlayManager::getElement(const std::string& elementId) {
    auto it = elements_.find(elementId);
    return (it != elements_.end()) ? &it->second : nullptr;
}

cv::Mat OverlayManager::renderOverlay(cv::Size outputSize) {
    // Create output buffer
    cv::Mat output = cv::Mat::zeros(outputSize, CV_8UC4);
    
    // Render each visible element
    for (const auto& pair : elements_) {
        const auto& element = pair.second;
        if (!element.visible) continue;
        
        switch (element.type) {
            case ElementType::TEXT:
                renderTextElement(output, element);
                break;
            case ElementType::IMAGE:
                renderImageElement(output, element);
                break;
            case ElementType::CUSTOM:
                renderChartElement(output, element);
                break;
            case ElementType::PLAYER_NAME:
                renderPlayerInfoElement(output, element);
                break;
            case ElementType::GAME_STATE:
                renderGameStatsElement(output, element);
                break;
            case ElementType::TIMER:
                renderTimerElement(output, element);
                break;
            case ElementType::SCORE:
                renderScoreElement(output, element);
                break;
        }
    }
    
    // Draw editor elements if in editor mode
    if (editorMode_) {
        drawElementBorders(output);
    }
    
    return output;
}

size_t OverlayManager::getMemoryUsage() const {
    size_t usage = sizeof(*this);
    usage += elements_.size() * sizeof(OverlayElement);
    usage += overlayBuffer_.total() * overlayBuffer_.elemSize();
    
    for (const auto& texture : elementTextures_) {
        usage += texture.second.total() * texture.second.elemSize();
    }
    
    lastMemoryUsage_ = usage;
    return usage;
}

void OverlayManager::optimizeForPerformance() {
    // Clear unused textures
    elementTextures_.clear();
    
    // Compact element storage
    auto now = std::chrono::steady_clock::now();
    lastOptimization_ = now;
}

void OverlayManager::setElementAnimation(const std::string& elementId, const std::string& animationType, float duration) {
    auto it = elements_.find(elementId);
    if (it == elements_.end()) return;
    
    ElementAnimation anim;
    anim.type = animationType;
    anim.duration = duration;
    anim.elapsed = 0.0f;
    anim.startPos = it->second.position;
    anim.startSize = it->second.size;
    anim.startOpacity = 1.0f;
    
    // Set target values based on animation type
    if (animationType == "fade_in") {
        anim.startOpacity = 0.0f;
        anim.targetOpacity = 1.0f;
    } else if (animationType == "slide_in") {
        anim.targetPos = it->second.position;
        anim.startPos.x -= 100.0f; // Slide from left
    }
    
    animations_[elementId] = anim;
}

void OverlayManager::updateAnimations(float deltaTime) {
    for (auto it = animations_.begin(); it != animations_.end();) {
        auto& anim = it->second;
        anim.elapsed += deltaTime;
        
        float progress = std::min(anim.elapsed / anim.duration, 1.0f);
        
        // Update element based on animation
        auto elementIt = elements_.find(it->first);
        if (elementIt != elements_.end()) {
            if (anim.type == "fade_in") {
                float opacity = anim.startOpacity + (anim.targetOpacity - anim.startOpacity) * progress;
                elementIt->second.properties["opacity"] = std::to_string(opacity);
            } else if (anim.type == "slide_in") {
                cv::Point2f pos = anim.startPos + (anim.targetPos - anim.startPos) * progress;
                elementIt->second.position = pos;
            }
        }
        
        // Remove completed animations
        if (progress >= 1.0f) {
            it = animations_.erase(it);
        } else {
            ++it;
        }
    }
}

// Rendering method implementations (simplified)
void OverlayManager::renderTextElement(cv::Mat& output, const OverlayElement& element) {
    std::string text = "Sample Text";
    auto it = element.properties.find("text");
    if (it != element.properties.end()) {
        text = it->second;
    }
    
    drawTextWithOutline(output, text, element.position, currentTemplate_.style);
}

void OverlayManager::renderImageElement(cv::Mat& output, const OverlayElement& element) {
    // Placeholder for image rendering
    cv::Rect rect(static_cast<int>(element.position.x), static_cast<int>(element.position.y), 
                  static_cast<int>(element.size.width), static_cast<int>(element.size.height));
    if (rect.x >= 0 && rect.y >= 0 && rect.x + rect.width < output.cols && rect.y + rect.height < output.rows) {
        cv::rectangle(output, rect, currentTemplate_.style.primaryColor, -1);
    }
}

void OverlayManager::renderChartElement(cv::Mat& output, const OverlayElement& element) {
    // Placeholder for chart rendering
    cv::Rect rect(static_cast<int>(element.position.x), static_cast<int>(element.position.y), 
                  static_cast<int>(element.size.width), static_cast<int>(element.size.height));
    if (rect.x >= 0 && rect.y >= 0 && rect.x + rect.width < output.cols && rect.y + rect.height < output.rows) {
        cv::rectangle(output, rect, currentTemplate_.style.secondaryColor, 2);
    }
}

void OverlayManager::renderProgressElement(cv::Mat& output, const OverlayElement& element) {
    cv::Rect rect(static_cast<int>(element.position.x), static_cast<int>(element.position.y), 
                  static_cast<int>(element.size.width), static_cast<int>(element.size.height));
    if (rect.x >= 0 && rect.y >= 0 && rect.x + rect.width < output.cols && rect.y + rect.height < output.rows) {
        // Background
        cv::rectangle(output, rect, currentTemplate_.style.backgroundColor, -1);
        // Progress bar
        int progressWidth = static_cast<int>(rect.width * currentData_.gameProgress);
        cv::Rect progressRect(rect.x, rect.y, progressWidth, rect.height);
        cv::rectangle(output, progressRect, currentTemplate_.style.primaryColor, -1);
    }
}

void OverlayManager::renderPlayerInfoElement(cv::Mat& output, const OverlayElement& element) {
    bool isPlayer1 = element.properties.find("player") == element.properties.end() || 
                     element.properties.at("player") == "1";
    
    std::string playerName;
    if (isPlayer1) {
        playerName = currentData_.player1Name;
    } else {
        playerName = currentData_.player2Name;
    }
    
    drawTextWithOutline(output, playerName, element.position, currentTemplate_.style);
}

void OverlayManager::renderGameStatsElement(cv::Mat& output, const OverlayElement& element) {
    std::string stats = "Game: " + currentData_.gameType + " | Rack: " + std::to_string(currentData_.currentRack);
    drawTextWithOutline(output, stats, element.position, currentTemplate_.style);
}

void OverlayManager::renderTimerElement(cv::Mat& output, const OverlayElement& element) {
    std::string time = formatTime(currentData_.gameStartTime);
    drawTextWithOutline(output, time, element.position, currentTemplate_.style);
}

void OverlayManager::renderScoreElement(cv::Mat& output, const OverlayElement& element) {
    std::string score = std::to_string(currentData_.player1Score) + " - " + std::to_string(currentData_.player2Score);
    drawTextWithOutline(output, score, element.position, currentTemplate_.style);
}

// Helper method implementations
void OverlayManager::applyGlassMorphismEffect(cv::Mat& region, const TemplateStyle& style) {
    if (style.glassMorphism) {
        cv::GaussianBlur(region, region, cv::Size(15, 15), 0);
        cv::addWeighted(region, 0.7, region, 0.3, 0, region);
    }
}

cv::Scalar OverlayManager::interpolateColor(cv::Scalar color1, cv::Scalar color2, float t) {
    return color1 * (1.0f - t) + color2 * t;
}

std::string OverlayManager::formatTime(std::chrono::steady_clock::time_point startTime) {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - startTime);
    
    int minutes = static_cast<int>(duration.count()) / 60;
    int seconds = static_cast<int>(duration.count()) % 60;
    
    return std::to_string(minutes) + ":" + (seconds < 10 ? "0" : "") + std::to_string(seconds);
}

void OverlayManager::drawTextWithOutline(cv::Mat& img, const std::string& text, cv::Point2f position, 
                                        const TemplateStyle& style, float scale) {
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 2, &baseline);
    
    cv::Point textPos(static_cast<int>(position.x), static_cast<int>(position.y + textSize.height));
    
    // Draw outline
    cv::putText(img, text, textPos, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 4);
    // Draw text
    cv::putText(img, text, textPos, cv::FONT_HERSHEY_SIMPLEX, scale, style.secondaryColor, 2);
}

cv::Rect2f OverlayManager::getElementBounds(const OverlayElement& element) {
    return cv::Rect2f(element.position.x, element.position.y, element.size.width, element.size.height);
}

void OverlayManager::drawElementBorders(cv::Mat& output) {
    for (const auto& pair : elements_) {
        const auto& element = pair.second;
        cv::Rect rect(static_cast<int>(element.position.x), static_cast<int>(element.position.y), 
                      static_cast<int>(element.size.width), static_cast<int>(element.size.height));
        
        if (rect.x >= 0 && rect.y >= 0 && rect.x + rect.width < output.cols && rect.y + rect.height < output.rows) {
            cv::rectangle(output, rect, cv::Scalar(255, 255, 0), 2); // Yellow border
            drawResizeHandles(output, element);
        }
    }
}

void OverlayManager::drawResizeHandles(cv::Mat& output, const OverlayElement& element) {
    cv::Point2f corners[4] = {
        element.position,
        cv::Point2f(element.position.x + element.size.width, element.position.y),
        cv::Point2f(element.position.x + element.size.width, element.position.y + element.size.height),
        cv::Point2f(element.position.x, element.position.y + element.size.height)
    };
    
    for (int i = 0; i < 4; ++i) {
        cv::circle(output, cv::Point(static_cast<int>(corners[i].x), static_cast<int>(corners[i].y)), 
                   5, cv::Scalar(255, 255, 0), -1);
    }
}

} // namespace streaming
} // namespace pv