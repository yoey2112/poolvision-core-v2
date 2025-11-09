#pragma once
#include <vector>
#include <map>
#include <optional>
#include <opencv2/opencv.hpp>
#include "StreamingTypes.hpp"

namespace pv {
namespace streaming {

/**
 * Manages overlay templates for streaming
 */
class TemplateSystem {
public:
    TemplateSystem();
    ~TemplateSystem();
    
    // Preset templates
    bool loadPresetTemplates();
    std::vector<OverlayTemplate> getPresetTemplates() const;
    
    // Template operations
    std::optional<OverlayTemplate> loadTemplate(const std::string& templateId);
    bool saveTemplate(const OverlayTemplate& overlayTemplate);
    bool deleteTemplate(const std::string& templateId);
    
    // Template creation helpers
    OverlayTemplate createClassicTournamentTemplate();
    OverlayTemplate createCasualGamingTemplate();
    OverlayTemplate createMinimalistTemplate();
    OverlayTemplate createEducationalTemplate();
    OverlayTemplate createSocialTemplate();
    
    // Template validation
    bool validateTemplate(const OverlayTemplate& overlayTemplate);
    std::vector<std::string> getTemplateErrors(const OverlayTemplate& overlayTemplate);

private:
    std::map<std::string, OverlayTemplate> presetTemplates_;
    std::map<std::string, OverlayTemplate> customTemplates_;
    std::string templatesPath_;
    
    // Helper methods for template creation
    OverlayElement createPlayerNameElement(const std::string& id, cv::Point2f position, bool isPlayer1);
    OverlayElement createScoreElement(const std::string& id, cv::Point2f position, bool isPlayer1);
    OverlayElement createGameProgressElement(cv::Point2f position);
    OverlayElement createTimerElement(cv::Point2f position);
    OverlayElement createGameTypeElement(cv::Point2f position);
    OverlayElement createRackNumberElement(cv::Point2f position);
    OverlayElement createShotSuggestionElement(cv::Point2f position);
    
    // Template I/O
    bool saveTemplateToFile(const OverlayTemplate& overlayTemplate);
    std::optional<OverlayTemplate> loadTemplateFromFile(const std::string& templateId);
    
    // Validation helpers
    bool isValidElementId(const std::string& id);
    bool isValidPosition(cv::Point2f position, cv::Size2f size);
    bool hasRequiredElements(const OverlayTemplate& overlayTemplate);
};

} // namespace streaming
} // namespace pv