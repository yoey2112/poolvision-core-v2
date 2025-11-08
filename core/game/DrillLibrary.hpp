#pragma once
#include "DrillSystem.hpp"
#include <vector>
#include <map>
#include <memory>

namespace pv {

/**
 * @brief Library of predefined and custom drills
 * 
 * Manages a collection of drill templates, provides drill discovery,
 * and handles custom drill creation and management.
 */
class DrillLibrary {
public:
    /**
     * @brief Drill template for creating variations
     */
    struct DrillTemplate {
        std::string name;
        std::string description;
        DrillSystem::Category category;
        DrillSystem::Difficulty baseDifficulty;
        std::function<DrillSystem::Drill(DrillSystem::Difficulty)> generator;
    };

    /**
     * @brief Construct a new Drill Library
     */
    DrillLibrary();
    ~DrillLibrary() = default;

    // Drill access
    /**
     * @brief Get drill by ID
     */
    const DrillSystem::Drill* getDrill(int drillId) const;
    
    /**
     * @brief Get all available drills
     */
    std::vector<DrillSystem::Drill> getAllDrills() const;
    
    /**
     * @brief Get all drill IDs
     */
    std::vector<int> getAllDrillIds() const;
    
    /**
     * @brief Get drills by category
     */
    std::vector<DrillSystem::Drill> getDrillsByCategory(DrillSystem::Category category) const;
    
    /**
     * @brief Get drills by difficulty
     */
    std::vector<DrillSystem::Drill> getDrillsByDifficulty(DrillSystem::Difficulty difficulty) const;
    
    /**
     * @brief Search drills by name or description
     */
    std::vector<DrillSystem::Drill> searchDrills(const std::string& query) const;

    // Custom drill management
    /**
     * @brief Create a new custom drill
     */
    int createCustomDrill(const DrillSystem::Drill& drill);
    
    /**
     * @brief Update existing custom drill
     */
    bool updateCustomDrill(int drillId, const DrillSystem::Drill& drill);
    
    /**
     * @brief Delete custom drill
     */
    bool deleteCustomDrill(int drillId);
    
    /**
     * @brief Get all custom drills
     */
    std::vector<DrillSystem::Drill> getCustomDrills() const;

    // Drill templates
    /**
     * @brief Get available drill templates
     */
    std::vector<DrillTemplate> getTemplates() const;
    
    /**
     * @brief Create drill from template
     */
    DrillSystem::Drill createFromTemplate(const std::string& templateName, 
                                         DrillSystem::Difficulty difficulty) const;

    // Predefined drill categories
    /**
     * @brief Get breaking drills
     */
    std::vector<DrillSystem::Drill> getBreakingDrills() const;
    
    /**
     * @brief Get cut shot drills
     */
    std::vector<DrillSystem::Drill> getCutShotDrills() const;
    
    /**
     * @brief Get bank shot drills
     */
    std::vector<DrillSystem::Drill> getBankShotDrills() const;
    
    /**
     * @brief Get position play drills
     */
    std::vector<DrillSystem::Drill> getPositionPlayDrills() const;
    
    /**
     * @brief Get speed control drills
     */
    std::vector<DrillSystem::Drill> getSpeedControlDrills() const;
    
    /**
     * @brief Get run-out drills
     */
    std::vector<DrillSystem::Drill> getRunOutDrills() const;

    // Utility functions
    /**
     * @brief Get next available drill ID
     */
    int getNextDrillId() const;
    
    /**
     * @brief Validate drill definition
     */
    bool validateDrill(const DrillSystem::Drill& drill) const;

private:
    std::map<int, DrillSystem::Drill> drills_;
    std::vector<DrillTemplate> templates_;
    int nextId_;

    // Drill creation helpers
    void initializePredefinedDrills();
    void initializeDrillTemplates();
    
    // Specific drill creators
    DrillSystem::Drill createStraightInDrill(DrillSystem::Difficulty difficulty);
    DrillSystem::Drill createCutShotDrill(DrillSystem::Difficulty difficulty);
    DrillSystem::Drill createBankShotDrill(DrillSystem::Difficulty difficulty);
    DrillSystem::Drill createPositionDrill(DrillSystem::Difficulty difficulty);
    DrillSystem::Drill createSpeedControlDrill(DrillSystem::Difficulty difficulty);
    DrillSystem::Drill createBreakingDrill(DrillSystem::Difficulty difficulty);
    DrillSystem::Drill createRunOutDrill(DrillSystem::Difficulty difficulty);
    DrillSystem::Drill createRailShotDrill(DrillSystem::Difficulty difficulty);
    DrillSystem::Drill createCombinationDrill(DrillSystem::Difficulty difficulty);
    DrillSystem::Drill createSafetyDrill(DrillSystem::Difficulty difficulty);
    
    // Helper functions
    std::vector<Ball> createStandardRack();
    std::vector<Ball> createPartialRack(const std::vector<int>& ballNumbers);
    cv::Point2f getTableCenter();
    cv::Point2f getRandomTablePosition();
    std::vector<cv::Point2f> getPocketPositions();
};

} // namespace pv