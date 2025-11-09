#include "TemplateSystem.hpp"
#include "StreamingEngine.hpp"
#include <iostream>
#include <fstream>

namespace pv {
namespace streaming {

TemplateSystem::TemplateSystem() : templatesPath_("templates/") {
    // Initialize with default empty maps
}

TemplateSystem::~TemplateSystem() = default;

bool TemplateSystem::loadPresetTemplates() {
    try {
        // Create the 5 preset templates
        presetTemplates_["classic_tournament"] = createClassicTournamentTemplate();
        presetTemplates_["casual_gaming"] = createCasualGamingTemplate();
        presetTemplates_["minimalist"] = createMinimalistTemplate();
        presetTemplates_["educational"] = createEducationalTemplate();
        presetTemplates_["social"] = createSocialTemplate();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load preset templates: " << e.what() << std::endl;
        return false;
    }
}

std::vector<OverlayTemplate> TemplateSystem::getPresetTemplates() const {
    std::vector<OverlayTemplate> templates;
    templates.reserve(presetTemplates_.size());
    
    for (const auto& pair : presetTemplates_) {
        templates.push_back(pair.second);
    }
    
    return templates;
}

std::optional<OverlayTemplate> TemplateSystem::loadTemplate(const std::string& templateId) {
    // First check preset templates
    auto presetIt = presetTemplates_.find(templateId);
    if (presetIt != presetTemplates_.end()) {
        return presetIt->second;
    }
    
    // Then check custom templates
    auto customIt = customTemplates_.find(templateId);
    if (customIt != customTemplates_.end()) {
        return customIt->second;
    }
    
    // Try loading from file
    return loadTemplateFromFile(templateId);
}

bool TemplateSystem::saveTemplate(const OverlayTemplate& overlayTemplate) {
    if (!validateTemplate(overlayTemplate)) {
        return false;
    }
    
    // Save to custom templates
    customTemplates_[overlayTemplate.id] = overlayTemplate;
    
    // Persist to file
    return saveTemplateToFile(overlayTemplate);
}

bool TemplateSystem::deleteTemplate(const std::string& templateId) {
    // Cannot delete preset templates
    if (presetTemplates_.find(templateId) != presetTemplates_.end()) {
        return false;
    }
    
    // Remove from custom templates
    auto it = customTemplates_.find(templateId);
    if (it != customTemplates_.end()) {
        customTemplates_.erase(it);
        return true;
    }
    
    return false;
}

OverlayTemplate TemplateSystem::createClassicTournamentTemplate() {
    OverlayTemplate template_("classic_tournament", "Classic Tournament");
    template_.description = "Professional tournament style overlay with complete game information";
    template_.author = "Pool Vision";
    
    // Style
    template_.style.primaryColor = cv::Scalar(255, 215, 0);     // Gold
    template_.style.secondaryColor = cv::Scalar(255, 255, 255); // White
    template_.style.backgroundColor = cv::Scalar(0, 0, 0, 180); // Semi-transparent black
    template_.style.fontFamily = "Arial";
    template_.style.fontSize = 18;
    template_.style.glassMorphism = true;
    
    // Elements
    template_.elements.push_back(createPlayerNameElement("player1_name", cv::Point2f(50, 50), true));
    template_.elements.push_back(createPlayerNameElement("player2_name", cv::Point2f(1400, 50), false));
    template_.elements.push_back(createScoreElement("score", cv::Point2f(720, 50), true));
    template_.elements.push_back(createGameProgressElement(cv::Point2f(500, 100)));
    template_.elements.push_back(createTimerElement(cv::Point2f(50, 100)));
    template_.elements.push_back(createGameTypeElement(cv::Point2f(50, 150)));
    template_.elements.push_back(createRackNumberElement(cv::Point2f(200, 150)));
    template_.elements.push_back(createShotSuggestionElement(cv::Point2f(500, 150)));
    
    return template_;
}

OverlayTemplate TemplateSystem::createCasualGamingTemplate() {
    OverlayTemplate template_("casual_gaming", "Casual Gaming");
    template_.description = "Fun and colorful overlay for casual streaming";
    template_.author = "Pool Vision";
    
    // Style
    template_.style.primaryColor = cv::Scalar(255, 0, 255);     // Magenta
    template_.style.secondaryColor = cv::Scalar(0, 255, 255);   // Cyan
    template_.style.backgroundColor = cv::Scalar(128, 0, 128, 160); // Purple semi-transparent
    template_.style.fontFamily = "Arial";
    template_.style.fontSize = 16;
    template_.style.glassMorphism = true;
    
    // Elements (simplified layout)
    template_.elements.push_back(createPlayerNameElement("player1_name", cv::Point2f(100, 80), true));
    template_.elements.push_back(createPlayerNameElement("player2_name", cv::Point2f(1300, 80), false));
    template_.elements.push_back(createScoreElement("score", cv::Point2f(750, 80), true));
    template_.elements.push_back(createTimerElement(cv::Point2f(100, 130)));
    template_.elements.push_back(createGameTypeElement(cv::Point2f(300, 130)));
    
    return template_;
}

OverlayTemplate TemplateSystem::createMinimalistTemplate() {
    OverlayTemplate template_("minimalist", "Minimalist");
    template_.description = "Clean and simple information display";
    template_.author = "Pool Vision";
    
    // Style
    template_.style.primaryColor = cv::Scalar(255, 255, 255);   // White
    template_.style.secondaryColor = cv::Scalar(200, 200, 200); // Light gray
    template_.style.backgroundColor = cv::Scalar(0, 0, 0, 100); // Very transparent black
    template_.style.fontFamily = "Arial";
    template_.style.fontSize = 14;
    template_.style.glassMorphism = false;
    
    // Elements (minimal set)
    template_.elements.push_back(createScoreElement("score", cv::Point2f(860, 50), true));
    template_.elements.push_back(createTimerElement(cv::Point2f(860, 80)));
    
    return template_;
}

OverlayTemplate TemplateSystem::createEducationalTemplate() {
    OverlayTemplate template_("educational", "Educational");
    template_.description = "Focus on learning with detailed game analysis";
    template_.author = "Pool Vision";
    
    // Style
    template_.style.primaryColor = cv::Scalar(0, 255, 0);       // Green
    template_.style.secondaryColor = cv::Scalar(255, 255, 255); // White
    template_.style.backgroundColor = cv::Scalar(0, 100, 0, 180); // Green semi-transparent
    template_.style.fontFamily = "Arial";
    template_.style.fontSize = 16;
    template_.style.glassMorphism = true;
    
    // Elements (educational focus)
    template_.elements.push_back(createPlayerNameElement("player1_name", cv::Point2f(50, 50), true));
    template_.elements.push_back(createPlayerNameElement("player2_name", cv::Point2f(1400, 50), false));
    template_.elements.push_back(createScoreElement("score", cv::Point2f(720, 50), true));
    template_.elements.push_back(createShotSuggestionElement(cv::Point2f(50, 100)));
    template_.elements.push_back(createGameProgressElement(cv::Point2f(400, 100)));
    template_.elements.push_back(createGameTypeElement(cv::Point2f(50, 150)));
    template_.elements.push_back(createTimerElement(cv::Point2f(200, 150)));
    
    return template_;
}

OverlayTemplate TemplateSystem::createSocialTemplate() {
    OverlayTemplate template_("social", "Social");
    template_.description = "Emphasize viewer engagement and social features";
    template_.author = "Pool Vision";
    
    // Style
    template_.style.primaryColor = cv::Scalar(255, 165, 0);     // Orange
    template_.style.secondaryColor = cv::Scalar(255, 255, 255); // White
    template_.style.backgroundColor = cv::Scalar(255, 100, 0, 180); // Orange semi-transparent
    template_.style.fontFamily = "Arial";
    template_.style.fontSize = 18;
    template_.style.glassMorphism = true;
    
    // Elements (social focus)
    template_.elements.push_back(createPlayerNameElement("player1_name", cv::Point2f(100, 60), true));
    template_.elements.push_back(createPlayerNameElement("player2_name", cv::Point2f(1350, 60), false));
    template_.elements.push_back(createScoreElement("score", cv::Point2f(720, 60), true));
    template_.elements.push_back(createTimerElement(cv::Point2f(100, 110)));
    template_.elements.push_back(createGameTypeElement(cv::Point2f(300, 110)));
    template_.elements.push_back(createRackNumberElement(cv::Point2f(500, 110)));
    
    return template_;
}

bool TemplateSystem::validateTemplate(const OverlayTemplate& overlayTemplate) {
    // Check required fields
    if (overlayTemplate.id.empty() || overlayTemplate.name.empty()) {
        return false;
    }
    
    // Validate elements
    for (const auto& element : overlayTemplate.elements) {
        if (element.id.empty() || !isValidElementId(element.id)) {
            return false;
        }
        
        if (!isValidPosition(element.position, element.size)) {
            return false;
        }
    }
    
    return hasRequiredElements(overlayTemplate);
}

std::vector<std::string> TemplateSystem::getTemplateErrors(const OverlayTemplate& overlayTemplate) {
    std::vector<std::string> errors;
    
    if (overlayTemplate.id.empty()) {
        errors.push_back("Template ID is required");
    }
    
    if (overlayTemplate.name.empty()) {
        errors.push_back("Template name is required");
    }
    
    for (const auto& element : overlayTemplate.elements) {
        if (element.id.empty()) {
            errors.push_back("Element ID is required");
        }
        
        if (!isValidElementId(element.id)) {
            errors.push_back("Invalid element ID: " + element.id);
        }
        
        if (!isValidPosition(element.position, element.size)) {
            errors.push_back("Invalid position/size for element: " + element.id);
        }
    }
    
    if (!hasRequiredElements(overlayTemplate)) {
        errors.push_back("Template missing required elements");
    }
    
    return errors;
}

// Helper method implementations
OverlayElement TemplateSystem::createPlayerNameElement(const std::string& id, cv::Point2f position, bool isPlayer1) {
    OverlayElement element(id, ElementType::PLAYER_NAME, position, cv::Size2f(200, 30));
    element.properties["player"] = isPlayer1 ? "1" : "2";
    element.properties["type"] = "name";
    return element;
}

OverlayElement TemplateSystem::createScoreElement(const std::string& id, cv::Point2f position, bool isPlayer1) {
    OverlayElement element(id, ElementType::SCORE, position, cv::Size2f(100, 30));
    element.properties["format"] = "P1 - P2";
    return element;
}

OverlayElement TemplateSystem::createGameProgressElement(cv::Point2f position) {
    OverlayElement element("game_progress", ElementType::CUSTOM, position, cv::Size2f(300, 20));
    element.properties["type"] = "game_progress";
    element.properties["max"] = "100";
    return element;
}

OverlayElement TemplateSystem::createTimerElement(cv::Point2f position) {
    OverlayElement element("timer", ElementType::TIMER, position, cv::Size2f(100, 30));
    element.properties["format"] = "MM:SS";
    return element;
}

OverlayElement TemplateSystem::createGameTypeElement(cv::Point2f position) {
    OverlayElement element("game_type", ElementType::TEXT, position, cv::Size2f(150, 30));
    element.properties["text"] = "8-Ball";
    return element;
}

OverlayElement TemplateSystem::createRackNumberElement(cv::Point2f position) {
    OverlayElement element("rack_number", ElementType::TEXT, position, cv::Size2f(100, 30));
    element.properties["text"] = "Rack 1";
    return element;
}

OverlayElement TemplateSystem::createShotSuggestionElement(cv::Point2f position) {
    OverlayElement element("shot_suggestion", ElementType::TEXT, position, cv::Size2f(300, 30));
    element.properties["text"] = "Shot Suggestion: None";
    return element;
}

bool TemplateSystem::saveTemplateToFile(const OverlayTemplate& overlayTemplate) {
    // Placeholder implementation - would save to JSON file
    std::cout << "Template saved: " << overlayTemplate.id << std::endl;
    return true;
}

std::optional<OverlayTemplate> TemplateSystem::loadTemplateFromFile(const std::string& templateId) {
    // Placeholder implementation - would load from JSON file
    return std::nullopt;
}

bool TemplateSystem::isValidElementId(const std::string& id) {
    // Check for valid characters and length
    if (id.empty() || id.length() > 100) {
        return false;
    }
    
    for (char c : id) {
        if (!std::isalnum(c) && c != '_' && c != '-') {
            return false;
        }
    }
    
    return true;
}

bool TemplateSystem::isValidPosition(cv::Point2f position, cv::Size2f size) {
    // Check bounds (assuming 1920x1080 max)
    return position.x >= 0 && position.y >= 0 && 
           position.x + size.width <= 1920 && 
           position.y + size.height <= 1080 &&
           size.width > 0 && size.height > 0;
}

bool TemplateSystem::hasRequiredElements(const OverlayTemplate& overlayTemplate) {
    // For now, any template is valid
    // Could enforce required elements like score, timer, etc.
    return !overlayTemplate.elements.empty();
}

} // namespace streaming
} // namespace pv