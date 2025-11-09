#include "DrillLibrary.hpp"
#include "../util/Types.hpp"
#include <algorithm>
#include <random>

namespace pv {

DrillLibrary::DrillLibrary() : nextId_(1) {
    initializePredefinedDrills();
    initializeDrillTemplates();
}

const DrillSystem::Drill* DrillLibrary::getDrill(int drillId) const {
    auto it = drills_.find(drillId);
    return (it != drills_.end()) ? &it->second : nullptr;
}

std::vector<DrillSystem::Drill> DrillLibrary::getAllDrills() const {
    std::vector<DrillSystem::Drill> drills;
    for (const auto& pair : drills_) {
        drills.push_back(pair.second);
    }
    return drills;
}

std::vector<int> DrillLibrary::getAllDrillIds() const {
    std::vector<int> ids;
    for (const auto& pair : drills_) {
        ids.push_back(pair.first);
    }
    return ids;
}

std::vector<DrillSystem::Drill> DrillLibrary::getDrillsByCategory(DrillSystem::Category category) const {
    std::vector<DrillSystem::Drill> drills;
    for (const auto& pair : drills_) {
        if (pair.second.category == category) {
            drills.push_back(pair.second);
        }
    }
    return drills;
}

std::vector<DrillSystem::Drill> DrillLibrary::getDrillsByDifficulty(DrillSystem::Difficulty difficulty) const {
    std::vector<DrillSystem::Drill> drills;
    for (const auto& pair : drills_) {
        if (pair.second.difficulty == difficulty) {
            drills.push_back(pair.second);
        }
    }
    return drills;
}

std::vector<DrillSystem::Drill> DrillLibrary::searchDrills(const std::string& query) const {
    std::vector<DrillSystem::Drill> results;
    std::string lowerQuery = query;
    std::transform(lowerQuery.begin(), lowerQuery.end(), lowerQuery.begin(), ::tolower);
    
    for (const auto& pair : drills_) {
        std::string lowerName = pair.second.name;
        std::string lowerDesc = pair.second.description;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
        std::transform(lowerDesc.begin(), lowerDesc.end(), lowerDesc.begin(), ::tolower);
        
        if (lowerName.find(lowerQuery) != std::string::npos ||
            lowerDesc.find(lowerQuery) != std::string::npos) {
            results.push_back(pair.second);
        }
    }
    
    return results;
}

int DrillLibrary::createCustomDrill(const DrillSystem::Drill& drill) {
    if (!validateDrill(drill)) {
        return 0;
    }
    
    DrillSystem::Drill newDrill = drill;
    newDrill.id = getNextDrillId();
    newDrill.isCustom = true;
    
    drills_[newDrill.id] = newDrill;
    return newDrill.id;
}

bool DrillLibrary::updateCustomDrill(int drillId, const DrillSystem::Drill& drill) {
    auto it = drills_.find(drillId);
    if (it == drills_.end() || !it->second.isCustom) {
        return false;
    }
    
    if (!validateDrill(drill)) {
        return false;
    }
    
    DrillSystem::Drill updatedDrill = drill;
    updatedDrill.id = drillId;
    updatedDrill.isCustom = true;
    
    drills_[drillId] = updatedDrill;
    return true;
}

bool DrillLibrary::deleteCustomDrill(int drillId) {
    auto it = drills_.find(drillId);
    if (it == drills_.end() || !it->second.isCustom) {
        return false;
    }
    
    drills_.erase(it);
    return true;
}

std::vector<DrillSystem::Drill> DrillLibrary::getCustomDrills() const {
    std::vector<DrillSystem::Drill> customDrills;
    for (const auto& pair : drills_) {
        if (pair.second.isCustom) {
            customDrills.push_back(pair.second);
        }
    }
    return customDrills;
}

std::vector<DrillLibrary::DrillTemplate> DrillLibrary::getTemplates() const {
    return templates_;
}

DrillSystem::Drill DrillLibrary::createFromTemplate(const std::string& templateName,
                                                   DrillSystem::Difficulty difficulty) const {
    for (const auto& tmpl : templates_) {
        if (tmpl.name == templateName) {
            return tmpl.generator(difficulty);
        }
    }
    
    return DrillSystem::Drill();
}

int DrillLibrary::getNextDrillId() const {
    int maxId = 0;
    for (const auto& pair : drills_) {
        maxId = std::max(maxId, pair.first);
    }
    return maxId + 1;
}

bool DrillLibrary::validateDrill(const DrillSystem::Drill& drill) const {
    if (drill.name.empty()) return false;
    if (drill.maxAttempts <= 0) return false;
    if (drill.successThreshold < 0.0 || drill.successThreshold > 1.0) return false;
    return true;
}

void DrillLibrary::initializePredefinedDrills() {
    // Create drills for each difficulty level
    for (int diff = 1; diff <= 5; ++diff) {
        auto difficulty = static_cast<DrillSystem::Difficulty>(diff);
        
        // Straight-in shots
        auto straightDrill = createStraightInDrill(difficulty);
        drills_[straightDrill.id] = straightDrill;
        
        // Cut shots
        auto cutDrill = createCutShotDrill(difficulty);
        drills_[cutDrill.id] = cutDrill;
        
        // Bank shots
        auto bankDrill = createBankShotDrill(difficulty);
        drills_[bankDrill.id] = bankDrill;
        
        // Position play
        auto positionDrill = createPositionDrill(difficulty);
        drills_[positionDrill.id] = positionDrill;
        
        // Speed control
        auto speedDrill = createSpeedControlDrill(difficulty);
        drills_[speedDrill.id] = speedDrill;
        
        // Breaking
        auto breakDrill = createBreakingDrill(difficulty);
        drills_[breakDrill.id] = breakDrill;
        
        // Run out
        auto runOutDrill = createRunOutDrill(difficulty);
        drills_[runOutDrill.id] = runOutDrill;
    }
}

void DrillLibrary::initializeDrillTemplates() {
    templates_.push_back({"Straight Shots", "Basic straight-in pocket shots", 
                         DrillSystem::Category::CutShots, DrillSystem::Difficulty::Beginner,
                         [this](DrillSystem::Difficulty diff) { return createStraightInDrill(diff); }});
    
    templates_.push_back({"Cut Shots", "Angled shots with varying difficulty", 
                         DrillSystem::Category::CutShots, DrillSystem::Difficulty::Intermediate,
                         [this](DrillSystem::Difficulty diff) { return createCutShotDrill(diff); }});
    
    templates_.push_back({"Bank Shots", "Shots requiring cushion rebounds", 
                         DrillSystem::Category::BankShots, DrillSystem::Difficulty::Advanced,
                         [this](DrillSystem::Difficulty diff) { return createBankShotDrill(diff); }});
    
    templates_.push_back({"Position Play", "Cue ball control exercises", 
                         DrillSystem::Category::PositionPlay, DrillSystem::Difficulty::Intermediate,
                         [this](DrillSystem::Difficulty diff) { return createPositionDrill(diff); }});
    
    templates_.push_back({"Speed Control", "Power and velocity management", 
                         DrillSystem::Category::SpeedControl, DrillSystem::Difficulty::Intermediate,
                         [this](DrillSystem::Difficulty diff) { return createSpeedControlDrill(diff); }});
}

DrillSystem::Drill DrillLibrary::createStraightInDrill(DrillSystem::Difficulty difficulty) {
    DrillSystem::Drill drill;
    drill.id = nextId_++;
    drill.name = "Straight In - " + DrillSystem::difficultyToString(difficulty);
    drill.description = "Practice straight pocket shots with increasing distance";
    drill.category = DrillSystem::Category::CutShots;
    drill.difficulty = difficulty;
    drill.maxAttempts = 10;
    drill.timeLimit = 0.0;
    drill.isCustom = false;
    
    // Set success threshold based on difficulty
    switch (difficulty) {
        case DrillSystem::Difficulty::Beginner:
            drill.successThreshold = 0.6;
            drill.instructions = "Shoot the object ball straight into the pocket. Focus on smooth stroke.";
            break;
        case DrillSystem::Difficulty::Intermediate:
            drill.successThreshold = 0.7;
            drill.instructions = "Straight shots with increased distance. Maintain accuracy.";
            break;
        case DrillSystem::Difficulty::Advanced:
            drill.successThreshold = 0.8;
            drill.instructions = "Long straight shots. Perfect alignment required.";
            break;
        case DrillSystem::Difficulty::Professional:
            drill.successThreshold = 0.9;
            drill.instructions = "Maximum distance straight shots with precision.";
            break;
        case DrillSystem::Difficulty::Expert:
            drill.successThreshold = 0.95;
            drill.instructions = "Perfect straight shots under pressure conditions.";
            break;
    }
    
    // Setup initial ball positions (cue ball and one object ball)
    Ball cueBall;
    cueBall.c.x = 200.0f;
    cueBall.c.y = 300.0f;
    cueBall.label = 0;
    drill.initialSetup.push_back(cueBall);
    
    Ball objectBall;
    objectBall.c.x = 400.0f + (static_cast<int>(difficulty) * 50);  // Increase distance with difficulty
    objectBall.c.y = 300.0f;
    objectBall.label = 1;
    drill.initialSetup.push_back(objectBall);
    
    // Target is the corner pocket
    drill.targets.push_back(cv::Point2f(600.0f, 50.0f));
    
    return drill;
}

DrillSystem::Drill DrillLibrary::createCutShotDrill(DrillSystem::Difficulty difficulty) {
    DrillSystem::Drill drill;
    drill.id = nextId_++;
    drill.name = "Cut Shots - " + DrillSystem::difficultyToString(difficulty);
    drill.description = "Practice angled shots with varying cut angles";
    drill.category = DrillSystem::Category::CutShots;
    drill.difficulty = difficulty;
    drill.maxAttempts = 10;
    drill.timeLimit = 0.0;
    drill.isCustom = false;
    
    switch (difficulty) {
        case DrillSystem::Difficulty::Beginner:
            drill.successThreshold = 0.5;
            drill.instructions = "Practice moderate cut angles. Focus on aim and follow-through.";
            break;
        case DrillSystem::Difficulty::Intermediate:
            drill.successThreshold = 0.6;
            drill.instructions = "Sharper cut angles. Maintain smooth stroke.";
            break;
        case DrillSystem::Difficulty::Advanced:
            drill.successThreshold = 0.7;
            drill.instructions = "Thin cuts and difficult angles.";
            break;
        case DrillSystem::Difficulty::Professional:
            drill.successThreshold = 0.8;
            drill.instructions = "Extreme cut shots with precision.";
            break;
        case DrillSystem::Difficulty::Expert:
            drill.successThreshold = 0.9;
            drill.instructions = "Maximum difficulty cuts under pressure.";
            break;
    }
    
    // Setup with angled ball position
    Ball cueBall;
    cueBall.c.x = 150.0f;
    cueBall.c.y = 250.0f;
    cueBall.label = 0;
    drill.initialSetup.push_back(cueBall);
    
    Ball objectBall;
    objectBall.c.x = 350.0f;
    objectBall.c.y = 200.0f + (static_cast<int>(difficulty) * 20);  // Vary angle with difficulty
    objectBall.label = 2;
    drill.initialSetup.push_back(objectBall);
    
    drill.targets.push_back(cv::Point2f(550.0f, 50.0f));
    
    return drill;
}

DrillSystem::Drill DrillLibrary::createBankShotDrill(DrillSystem::Difficulty difficulty) {
    DrillSystem::Drill drill;
    drill.id = nextId_++;
    drill.name = "Bank Shots - " + DrillSystem::difficultyToString(difficulty);
    drill.description = "Practice shots requiring cushion rebounds";
    drill.category = DrillSystem::Category::BankShots;
    drill.difficulty = difficulty;
    drill.maxAttempts = 15;  // More attempts for harder shots
    drill.timeLimit = 0.0;
    drill.isCustom = false;
    
    switch (difficulty) {
        case DrillSystem::Difficulty::Beginner:
            drill.successThreshold = 0.4;
            drill.instructions = "Simple one-rail bank shots. Focus on angle calculation.";
            break;
        case DrillSystem::Difficulty::Intermediate:
            drill.successThreshold = 0.5;
            drill.instructions = "Cross-table banks. Use diamonds for reference.";
            break;
        case DrillSystem::Difficulty::Advanced:
            drill.successThreshold = 0.6;
            drill.instructions = "Two-rail banks. Advanced angle geometry.";
            break;
        case DrillSystem::Difficulty::Professional:
            drill.successThreshold = 0.7;
            drill.instructions = "Multi-rail banks with precise calculation.";
            break;
        case DrillSystem::Difficulty::Expert:
            drill.successThreshold = 0.8;
            drill.instructions = "Complex bank patterns under tournament conditions.";
            break;
    }
    
    // Setup for bank shot
    Ball cueBall;
    cueBall.c.x = 100.0f;
    cueBall.c.y = 350.0f;
    cueBall.label = 0;
    drill.initialSetup.push_back(cueBall);
    
    Ball objectBall;
    objectBall.c.x = 300.0f;
    objectBall.c.y = 150.0f;
    objectBall.label = 3;
    drill.initialSetup.push_back(objectBall);
    
    drill.targets.push_back(cv::Point2f(50.0f, 350.0f));  // Opposite corner
    
    return drill;
}

DrillSystem::Drill DrillLibrary::createPositionDrill(DrillSystem::Difficulty difficulty) {
    DrillSystem::Drill drill;
    drill.id = nextId_++;
    drill.name = "Position Play - " + DrillSystem::difficultyToString(difficulty);
    drill.description = "Practice cue ball control and positioning";
    drill.category = DrillSystem::Category::PositionPlay;
    drill.difficulty = difficulty;
    drill.maxAttempts = 8;
    drill.timeLimit = 0.0;
    drill.isCustom = false;
    
    switch (difficulty) {
        case DrillSystem::Difficulty::Beginner:
            drill.successThreshold = 0.5;
            drill.instructions = "Pocket the ball and stop cue ball in target zone.";
            break;
        case DrillSystem::Difficulty::Intermediate:
            drill.successThreshold = 0.6;
            drill.instructions = "Control cue ball to specific position for next shot.";
            break;
        case DrillSystem::Difficulty::Advanced:
            drill.successThreshold = 0.7;
            drill.instructions = "Precise position play with spin control.";
            break;
        case DrillSystem::Difficulty::Professional:
            drill.successThreshold = 0.8;
            drill.instructions = "Perfect position for run-out sequence.";
            break;
        case DrillSystem::Difficulty::Expert:
            drill.successThreshold = 0.9;
            drill.instructions = "Master-level position control under pressure.";
            break;
    }
    
    // Setup with multiple balls for positioning practice
    Ball cueBall;
    cueBall.c.x = 200.0f;
    cueBall.c.y = 300.0f;
    cueBall.label = 0;
    drill.initialSetup.push_back(cueBall);
    
    Ball objectBall;
    objectBall.c.x = 350.0f;
    objectBall.c.y = 250.0f;
    objectBall.label = 4;
    drill.initialSetup.push_back(objectBall);
    
    drill.targets.push_back(cv::Point2f(450.0f, 200.0f));  // Position target
    
    return drill;
}

DrillSystem::Drill DrillLibrary::createSpeedControlDrill(DrillSystem::Difficulty difficulty) {
    DrillSystem::Drill drill;
    drill.id = nextId_++;
    drill.name = "Speed Control - " + DrillSystem::difficultyToString(difficulty);
    drill.description = "Practice power and velocity management";
    drill.category = DrillSystem::Category::SpeedControl;
    drill.difficulty = difficulty;
    drill.maxAttempts = 10;
    drill.timeLimit = 0.0;
    drill.isCustom = false;
    
    switch (difficulty) {
        case DrillSystem::Difficulty::Beginner:
            drill.successThreshold = 0.5;
            drill.instructions = "Control shot speed - soft, medium, and firm strokes.";
            break;
        case DrillSystem::Difficulty::Intermediate:
            drill.successThreshold = 0.6;
            drill.instructions = "Precise speed control for different distances.";
            break;
        case DrillSystem::Difficulty::Advanced:
            drill.successThreshold = 0.7;
            drill.instructions = "Fine speed adjustments with position requirements.";
            break;
        case DrillSystem::Difficulty::Professional:
            drill.successThreshold = 0.8;
            drill.instructions = "Perfect speed control for professional play.";
            break;
        case DrillSystem::Difficulty::Expert:
            drill.successThreshold = 0.9;
            drill.instructions = "Master-level speed precision under pressure.";
            break;
    }
    
    // Setup for speed control
    Ball cueBall;
    cueBall.c.x = 150.0f;
    cueBall.c.y = 300.0f;
    cueBall.label = 0;
    drill.initialSetup.push_back(cueBall);
    
    Ball objectBall;
    objectBall.c.x = 400.0f;
    objectBall.c.y = 300.0f;
    objectBall.label = 5;
    drill.initialSetup.push_back(objectBall);
    
    drill.targets.push_back(cv::Point2f(500.0f, 250.0f));
    
    return drill;
}

DrillSystem::Drill DrillLibrary::createBreakingDrill(DrillSystem::Difficulty difficulty) {
    DrillSystem::Drill drill;
    drill.id = nextId_++;
    drill.name = "Breaking - " + DrillSystem::difficultyToString(difficulty);
    drill.description = "Practice break shots and ball spread";
    drill.category = DrillSystem::Category::Breaking;
    drill.difficulty = difficulty;
    drill.maxAttempts = 5;
    drill.timeLimit = 0.0;
    drill.isCustom = false;
    
    switch (difficulty) {
        case DrillSystem::Difficulty::Beginner:
            drill.successThreshold = 0.4;
            drill.instructions = "Focus on solid contact and ball spread.";
            break;
        case DrillSystem::Difficulty::Intermediate:
            drill.successThreshold = 0.5;
            drill.instructions = "Controlled power break with cue ball control.";
            break;
        case DrillSystem::Difficulty::Advanced:
            drill.successThreshold = 0.6;
            drill.instructions = "Power break with specific cue ball placement.";
            break;
        case DrillSystem::Difficulty::Professional:
            drill.successThreshold = 0.7;
            drill.instructions = "Professional break with maximum effectiveness.";
            break;
        case DrillSystem::Difficulty::Expert:
            drill.successThreshold = 0.8;
            drill.instructions = "Perfect break technique under competition pressure.";
            break;
    }
    
    // Setup full rack
    drill.initialSetup = createStandardRack();
    
    drill.targets.push_back(cv::Point2f(300.0f, 300.0f));  // Center table for cue ball control
    
    return drill;
}

DrillSystem::Drill DrillLibrary::createRunOutDrill(DrillSystem::Difficulty difficulty) {
    DrillSystem::Drill drill;
    drill.id = nextId_++;
    drill.name = "Run Out - " + DrillSystem::difficultyToString(difficulty);
    drill.description = "Practice clearing remaining balls in sequence";
    drill.category = DrillSystem::Category::RunOut;
    drill.difficulty = difficulty;
    drill.maxAttempts = 3;  // Fewer attempts for run-out drills
    drill.timeLimit = 0.0;
    drill.isCustom = false;
    
    switch (difficulty) {
        case DrillSystem::Difficulty::Beginner:
            drill.successThreshold = 0.6;
            drill.instructions = "Clear 3 balls in sequence with position play.";
            break;
        case DrillSystem::Difficulty::Intermediate:
            drill.successThreshold = 0.7;
            drill.instructions = "Run out 5 balls with strategic positioning.";
            break;
        case DrillSystem::Difficulty::Advanced:
            drill.successThreshold = 0.8;
            drill.instructions = "Complete 7-ball run with perfect position.";
            break;
        case DrillSystem::Difficulty::Professional:
            drill.successThreshold = 0.85;
            drill.instructions = "Full rack run-out with professional technique.";
            break;
        case DrillSystem::Difficulty::Expert:
            drill.successThreshold = 0.9;
            drill.instructions = "Perfect run-out under tournament pressure.";
            break;
    }
    
    // Create partial rack based on difficulty
    std::vector<int> ballNumbers;
    int ballCount = 3 + static_cast<int>(difficulty);
    for (int i = 1; i <= ballCount; ++i) {
        ballNumbers.push_back(i);
    }
    drill.initialSetup = createPartialRack(ballNumbers);
    
    drill.targets.push_back(cv::Point2f(400.0f, 200.0f));  // Final position target
    
    return drill;
}

DrillSystem::Drill DrillLibrary::createRailShotDrill(DrillSystem::Difficulty difficulty) {
    DrillSystem::Drill drill;
    drill.id = nextId_++;
    drill.name = "Rail Shots - " + DrillSystem::difficultyToString(difficulty);
    drill.description = "Practice shots along the rail";
    drill.category = DrillSystem::Category::RailShots;
    drill.difficulty = difficulty;
    drill.maxAttempts = 12;
    drill.timeLimit = 0.0;
    drill.isCustom = false;
    
    // Rail shots setup would go here
    return drill;
}

DrillSystem::Drill DrillLibrary::createCombinationDrill(DrillSystem::Difficulty difficulty) {
    DrillSystem::Drill drill;
    drill.id = nextId_++;
    drill.name = "Combinations - " + DrillSystem::difficultyToString(difficulty);
    drill.description = "Practice combination shots";
    drill.category = DrillSystem::Category::Combinations;
    drill.difficulty = difficulty;
    drill.maxAttempts = 15;
    drill.timeLimit = 0.0;
    drill.isCustom = false;
    
    // Combination shots setup would go here
    return drill;
}

DrillSystem::Drill DrillLibrary::createSafetyDrill(DrillSystem::Difficulty difficulty) {
    DrillSystem::Drill drill;
    drill.id = nextId_++;
    drill.name = "Safety Play - " + DrillSystem::difficultyToString(difficulty);
    drill.description = "Practice defensive safety shots";
    drill.category = DrillSystem::Category::Safety;
    drill.difficulty = difficulty;
    drill.maxAttempts = 8;
    drill.timeLimit = 0.0;
    drill.isCustom = false;
    
    // Safety play setup would go here
    return drill;
}

std::vector<Ball> DrillLibrary::createStandardRack() {
    std::vector<Ball> balls;
    
    // Create standard 8-ball rack
    float rackX = 400.0f;
    float rackY = 200.0f;
    float ballDiameter = BALL_RADIUS * 2;
    
    // Standard rack positions
    std::vector<cv::Point2f> positions = {
        {rackX, rackY},  // 1-ball
        {rackX + ballDiameter, rackY - ballDiameter/2},  // Left 2nd row
        {rackX + ballDiameter, rackY + ballDiameter/2},  // Right 2nd row
        // ... continue for full rack
    };
    
    for (int i = 0; i < positions.size() && i < 15; ++i) {
        Ball ball;
        ball.c.x = positions[i].x;
        ball.c.y = positions[i].y;
        ball.label = i + 1;
        balls.push_back(ball);
    }
    
    return balls;
}

std::vector<Ball> DrillLibrary::createPartialRack(const std::vector<int>& ballNumbers) {
    std::vector<Ball> balls;
    auto standardRack = createStandardRack();
    
    for (int ballNum : ballNumbers) {
        if (ballNum > 0 && ballNum <= standardRack.size()) {
            balls.push_back(standardRack[ballNum - 1]);
        }
    }
    
    return balls;
}

cv::Point2f DrillLibrary::getTableCenter() {
    return cv::Point2f(300.0f, 200.0f);  // Approximate table center
}

cv::Point2f DrillLibrary::getRandomTablePosition() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> xDist(50.0f, 550.0f);
    std::uniform_real_distribution<float> yDist(50.0f, 350.0f);
    
    return cv::Point2f(xDist(gen), yDist(gen));
}

std::vector<cv::Point2f> DrillLibrary::getPocketPositions() {
    return {
        {50.0f, 50.0f},    // Top-left
        {300.0f, 50.0f},   // Top-center
        {550.0f, 50.0f},   // Top-right
        {50.0f, 350.0f},   // Bottom-left
        {300.0f, 350.0f},  // Bottom-center
        {550.0f, 350.0f}   // Bottom-right
    };
}

} // namespace pv
