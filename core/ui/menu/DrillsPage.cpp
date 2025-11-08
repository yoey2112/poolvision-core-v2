#include "DrillsPage.hpp"
#include "../../game/DrillLibrary.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace pv {

DrillsPage::DrillsPage(Database& database, std::shared_ptr<DrillSystem> drillSystem)
    : database_(database), drillSystem_(drillSystem), 
      currentState_(DrillUIState::DrillLibrary), activePlayerId_(1),
      selectedDrillId_(0), scrollOffset_(0), mousePressed_(false) {
    
    createButtons();
    updateFilteredDrills();
    
    // Set up drill system callbacks
    drillSystem_->setStateChangeCallback([this](DrillSystem::DrillState state) {
        // Update UI based on drill state changes
        if (state == DrillSystem::DrillState::Completed) {
            currentState_ = DrillUIState::DrillResults;
        }
    });
    
    drillSystem_->setAttemptCallback([this](const DrillSystem::DrillAttempt& attempt) {
        // Update execution state with new attempt
        executionState_.attempts.push_back(attempt);
        executionState_.currentAccuracy = attempt.accuracy;
        executionState_.feedback = attempt.feedback;
        if (attempt.accuracy > executionState_.bestAccuracy) {
            executionState_.bestAccuracy = attempt.accuracy;
        }
    });
}

void DrillsPage::render(cv::Mat& frame) {
    // Clear background with dark theme
    frame = cv::Scalar(26, 26, 26);  // DarkBg
    
    // Draw title bar
    UITheme::drawTitleBar(frame, "Practice Drills");
    
    switch (currentState_) {
        case DrillUIState::DrillLibrary:
            renderDrillLibrary(frame);
            break;
        case DrillUIState::DrillExecution:
            renderDrillExecution(frame);
            break;
        case DrillUIState::DrillResults:
            renderDrillResults(frame);
            break;
        case DrillUIState::DrillCreator:
            renderDrillCreator(frame);
            break;
        case DrillUIState::DrillStats:
            renderDrillStats(frame);
            break;
    }
    
    // Draw navigation buttons
    for (const auto& button : buttons_) {
        if (button.visible) {
            UITheme::drawButton(frame, button.label, button.rect, false, false, !button.enabled);
        }
    }
}

void DrillsPage::handleClick(const cv::Point& clickPos) {
    lastClickPos_ = clickPos;
    mousePressed_ = true;
    
    // Check button clicks first
    auto* button = findButtonAt(clickPos);
    if (button && button->enabled) {
        button->action();
        return;
    }
    
    // Handle state-specific clicks
    switch (currentState_) {
        case DrillUIState::DrillLibrary: {
            // Check drill card clicks
            cv::Rect listArea(FILTER_PANEL_WIDTH, 100, 
                             frame.cols - FILTER_PANEL_WIDTH - DETAIL_PANEL_WIDTH - 20, 
                             frame.rows - 200);
            
            if (UITheme::isPointInRect(clickPos, listArea)) {
                // Calculate which drill card was clicked
                int cardIndex = (clickPos.y - listArea.y + scrollOffset_) / (CARD_HEIGHT + CARD_MARGIN);
                if (cardIndex >= 0 && cardIndex < filteredDrills_.size()) {
                    selectedDrillId_ = filteredDrills_[cardIndex].id;
                }
            }
            break;
        }
        case DrillUIState::DrillExecution: {
            // Handle execution clicks (pause/resume/etc)
            break;
        }
        default:
            break;
    }
}

void DrillsPage::handleKeyPress(int key) {
    switch (key) {
        case 27:  // ESC
            if (currentState_ == DrillUIState::DrillExecution) {
                endCurrentDrill();
            } else {
                currentState_ = DrillUIState::DrillLibrary;
            }
            break;
        case 32:  // SPACE
            if (currentState_ == DrillUIState::DrillExecution) {
                if (executionState_.isPaused) {
                    resumeCurrentDrill();
                } else {
                    pauseCurrentDrill();
                }
            }
            break;
        case 13:  // ENTER
            if (currentState_ == DrillUIState::DrillLibrary && selectedDrillId_ > 0) {
                startSelectedDrill();
            }
            break;
        case 114: // 'r'
            if (currentState_ == DrillUIState::DrillExecution) {
                resetCurrentDrill();
            }
            break;
    }
}

void DrillsPage::update() {
    if (currentState_ == DrillUIState::DrillExecution && drillSystem_->isDrillActive()) {
        // Update execution state
        const auto& session = drillSystem_->getCurrentSession();
        executionState_.currentAttempt = session.currentAttempt;
        executionState_.totalAttempts = session.attempts.size();
        executionState_.isPaused = drillSystem_->isDrillPaused();
        
        // Update time remaining if drill has time limit
        auto drill = drillSystem_->getCurrentDrill();
        if (drill && drill->timeLimit > 0) {
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - session.startTime).count();
            executionState_.timeRemaining = std::max(0.0, drill->timeLimit - elapsed);
        }
    }
    
    updateButtons();
}

void DrillsPage::renderDrillLibrary(cv::Mat& frame) {
    // Define layout areas
    cv::Rect filterArea(10, 100, FILTER_PANEL_WIDTH, frame.rows - 200);
    cv::Rect listArea(FILTER_PANEL_WIDTH + 10, 100, 
                     frame.cols - FILTER_PANEL_WIDTH - DETAIL_PANEL_WIDTH - 30, 
                     frame.rows - 200);
    cv::Rect detailArea(frame.cols - DETAIL_PANEL_WIDTH - 10, 100, 
                       DETAIL_PANEL_WIDTH, frame.rows - 200);
    
    // Render components
    renderFilterPanel(frame, filterArea);
    renderDrillList(frame, listArea);
    
    // Render selected drill details
    if (selectedDrillId_ > 0) {
        auto it = std::find_if(filteredDrills_.begin(), filteredDrills_.end(),
                              [this](const DrillSystem::Drill& d) { return d.id == selectedDrillId_; });
        if (it != filteredDrills_.end()) {
            renderDrillDetails(frame, *it, detailArea);
        }
    }
}

void DrillsPage::renderDrillExecution(cv::Mat& frame) {
    // Define layout areas
    cv::Rect hudArea(10, 100, frame.cols - 20, 120);
    cv::Rect instructionArea(10, 230, frame.cols / 2 - 15, 150);
    cv::Rect historyArea(frame.cols / 2 + 5, 230, frame.cols / 2 - 15, 150);
    cv::Rect progressArea(10, 390, frame.cols - 20, 60);
    cv::Rect controlArea(10, 460, frame.cols - 20, 80);
    
    // Render components
    renderExecutionHUD(frame);
    renderDrillInstructions(frame, instructionArea);
    renderAttemptHistory(frame, historyArea);
    renderProgressBar(frame, progressArea);
    renderExecutionControls(frame, controlArea);
}

void DrillsPage::renderDrillResults(cv::Mat& frame) {
    cv::Rect summaryArea(10, 100, frame.cols - 20, 200);
    cv::Rect chartArea(10, 310, frame.cols / 2 - 15, 200);
    cv::Rect metricsArea(frame.cols / 2 + 5, 310, frame.cols / 2 - 15, 200);
    
    renderResultsSummary(frame, summaryArea);
    renderAccuracyChart(frame, chartArea);
    renderPerformanceMetrics(frame, metricsArea);
}

void DrillsPage::renderDrillCreator(cv::Mat& frame) {
    cv::Rect designArea(10, 100, frame.cols / 2 - 15, frame.rows - 200);
    cv::Rect tableArea(frame.cols / 2 + 5, 100, frame.cols / 2 - 15, 300);
    cv::Rect propertiesArea(frame.cols / 2 + 5, 410, frame.cols / 2 - 15, frame.rows - 510);
    
    renderDrillDesigner(frame, designArea);
    renderBallPlacement(frame, tableArea);
    renderDrillProperties(frame, propertiesArea);
}

void DrillsPage::renderDrillStats(cv::Mat& frame) {
    cv::Rect overviewArea(10, 100, frame.cols - 20, 150);
    cv::Rect progressArea(10, 260, frame.cols / 2 - 15, 200);
    cv::Rect trendsArea(frame.cols / 2 + 5, 260, frame.cols / 2 - 15, 200);
    
    renderStatsOverview(frame, overviewArea);
    renderDrillProgress(frame, progressArea);
    renderImprovementTrends(frame, trendsArea);
}

void DrillsPage::renderDrillList(cv::Mat& frame, const cv::Rect& listArea) {
    UITheme::drawCard(frame, listArea, UITheme::Colors::MediumBg);
    
    // Draw drill cards
    int yPos = listArea.y + 10 - scrollOffset_;
    
    for (size_t i = 0; i < filteredDrills_.size(); ++i) {
        const auto& drill = filteredDrills_[i];
        cv::Rect cardRect(listArea.x + 10, yPos, listArea.width - 20, CARD_HEIGHT);
        
        // Only render visible cards
        if (cardRect.y + cardRect.height > listArea.y && cardRect.y < listArea.y + listArea.height) {
            bool isSelected = (drill.id == selectedDrillId_);
            renderDrillCard(frame, drill, cardRect, isSelected);
        }
        
        yPos += CARD_HEIGHT + CARD_MARGIN;
    }
}

void DrillsPage::renderDrillCard(cv::Mat& frame, const DrillSystem::Drill& drill, 
                                const cv::Rect& cardRect, bool isSelected) {
    cv::Scalar bgColor = isSelected ? UITheme::Colors::ButtonActive : UITheme::Colors::LightBg;
    cv::Scalar textColor = UITheme::Colors::TextPrimary;
    
    UITheme::drawCard(frame, cardRect, bgColor);
    
    // Draw drill name
    cv::Point namePos(cardRect.x + 15, cardRect.y + 25);
    UITheme::drawTextWithShadow(frame, drill.name, namePos, 
                                UITheme::Fonts::FontFaceBold, UITheme::Fonts::HeadingSize,
                                textColor, UITheme::Fonts::HeadingThickness);
    
    // Draw category and difficulty
    std::string categoryText = DrillSystem::categoryToString(drill.category);
    std::string difficultyText = DrillSystem::difficultyToString(drill.difficulty);
    std::string infoText = categoryText + " • " + difficultyText;
    
    cv::Point infoPos(cardRect.x + 15, cardRect.y + 50);
    cv::putText(frame, infoText, infoPos, UITheme::Fonts::FontFace,
                UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                UITheme::Fonts::BodyThickness);
    
    // Draw description
    cv::Point descPos(cardRect.x + 15, cardRect.y + 75);
    cv::putText(frame, drill.description.substr(0, 60) + "...", descPos, 
                UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
                UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness);
    
    // Draw difficulty stars
    int stars = static_cast<int>(drill.difficulty);
    cv::Point starPos(cardRect.x + cardRect.width - 100, cardRect.y + 25);
    for (int i = 0; i < 5; ++i) {
        cv::Scalar starColor = (i < stars) ? UITheme::Colors::NeonYellow : UITheme::Colors::TextDisabled;
        cv::putText(frame, "★", cv::Point(starPos.x + i * 15, starPos.y), 
                   UITheme::Fonts::FontFace, 0.6, starColor, 2);
    }
    
    // Draw max attempts info
    std::string attemptsText = std::to_string(drill.maxAttempts) + " attempts";
    cv::Point attemptsPos(cardRect.x + cardRect.width - 120, cardRect.y + 95);
    cv::putText(frame, attemptsText, attemptsPos, UITheme::Fonts::FontFace,
                UITheme::Fonts::SmallSize, UITheme::Colors::TextSecondary,
                UITheme::Fonts::BodyThickness);
}

void DrillsPage::renderFilterPanel(cv::Mat& frame, const cv::Rect& filterArea) {
    UITheme::drawCard(frame, filterArea, UITheme::Colors::MediumBg);
    
    // Title
    cv::Point titlePos(filterArea.x + 15, filterArea.y + 25);
    UITheme::drawTextWithShadow(frame, "Filters", titlePos,
                                UITheme::Fonts::FontFaceBold, UITheme::Fonts::HeadingSize,
                                UITheme::Colors::TextPrimary, UITheme::Fonts::HeadingThickness);
    
    // Category filter
    int yOffset = 60;
    cv::Point catPos(filterArea.x + 15, filterArea.y + yOffset);
    cv::putText(frame, "Category:", catPos, UITheme::Fonts::FontFace,
                UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                UITheme::Fonts::BodyThickness);
    
    // Category dropdown
    cv::Rect catDropdown(filterArea.x + 15, filterArea.y + yOffset + 20, 200, 30);
    std::string categoryText = DrillSystem::categoryToString(currentFilter_.category);
    UITheme::drawDropdown(frame, categoryText, catDropdown);
    
    // Difficulty filter
    yOffset += 80;
    cv::Point diffPos(filterArea.x + 15, filterArea.y + yOffset);
    cv::putText(frame, "Difficulty:", diffPos, UITheme::Fonts::FontFace,
                UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                UITheme::Fonts::BodyThickness);
    
    // Difficulty dropdown
    cv::Rect diffDropdown(filterArea.x + 15, filterArea.y + yOffset + 20, 200, 30);
    std::string difficultyText = DrillSystem::difficultyToString(currentFilter_.difficulty);
    UITheme::drawDropdown(frame, difficultyText, diffDropdown);
    
    // Search box
    yOffset += 80;
    cv::Point searchPos(filterArea.x + 15, filterArea.y + yOffset);
    cv::putText(frame, "Search:", searchPos, UITheme::Fonts::FontFace,
                UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                UITheme::Fonts::BodyThickness);
    
    cv::Rect searchBox(filterArea.x + 15, filterArea.y + yOffset + 20, 200, 30);
    cv::rectangle(frame, searchBox, UITheme::Colors::LightBg, -1);
    cv::rectangle(frame, searchBox, UITheme::Colors::BorderColor, 1);
    
    if (!currentFilter_.searchQuery.empty()) {
        cv::Point textPos(searchBox.x + 5, searchBox.y + 20);
        cv::putText(frame, currentFilter_.searchQuery, textPos, UITheme::Fonts::FontFace,
                    UITheme::Fonts::BodySize, UITheme::Colors::TextPrimary,
                    UITheme::Fonts::BodyThickness);
    }
    
    // Filter toggles
    yOffset += 80;
    cv::Rect customToggle(filterArea.x + 15, filterArea.y + yOffset, 20, 20);
    UITheme::drawToggle(frame, currentFilter_.showCustomOnly, customToggle);
    cv::Point customLabel(customToggle.x + 30, customToggle.y + 15);
    cv::putText(frame, "Custom only", customLabel, UITheme::Fonts::FontFace,
                UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                UITheme::Fonts::BodyThickness);
    
    yOffset += 40;
    cv::Rect favToggle(filterArea.x + 15, filterArea.y + yOffset, 20, 20);
    UITheme::drawToggle(frame, currentFilter_.showFavoritesOnly, favToggle);
    cv::Point favLabel(favToggle.x + 30, favToggle.y + 15);
    cv::putText(frame, "Favorites only", favLabel, UITheme::Fonts::FontFace,
                UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                UITheme::Fonts::BodyThickness);
}

void DrillsPage::renderDrillDetails(cv::Mat& frame, const DrillSystem::Drill& drill, const cv::Rect& detailArea) {
    UITheme::drawCard(frame, detailArea, UITheme::Colors::MediumBg);
    
    int yOffset = 20;
    
    // Title
    cv::Point titlePos(detailArea.x + 15, detailArea.y + yOffset);
    UITheme::drawTextWithShadow(frame, drill.name, titlePos,
                                UITheme::Fonts::FontFaceBold, UITheme::Fonts::HeadingSize,
                                UITheme::Colors::TextPrimary, UITheme::Fonts::HeadingThickness);
    
    yOffset += 40;
    
    // Category and difficulty
    std::string categoryText = "Category: " + DrillSystem::categoryToString(drill.category);
    cv::Point catPos(detailArea.x + 15, detailArea.y + yOffset);
    cv::putText(frame, categoryText, catPos, UITheme::Fonts::FontFace,
                UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                UITheme::Fonts::BodyThickness);
    
    yOffset += 25;
    std::string diffText = "Difficulty: " + DrillSystem::difficultyToString(drill.difficulty);
    cv::Point diffPos(detailArea.x + 15, detailArea.y + yOffset);
    cv::putText(frame, diffText, diffPos, UITheme::Fonts::FontFace,
                UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                UITheme::Fonts::BodyThickness);
    
    yOffset += 40;
    
    // Description
    cv::Point descTitlePos(detailArea.x + 15, detailArea.y + yOffset);
    cv::putText(frame, "Description:", descTitlePos, UITheme::Fonts::FontFaceBold,
                UITheme::Fonts::BodySize, UITheme::Colors::TextPrimary,
                UITheme::Fonts::BodyThickness);
    
    yOffset += 25;
    cv::Point descPos(detailArea.x + 15, detailArea.y + yOffset);
    
    // Word wrap description
    std::istringstream words(drill.description);
    std::string word;
    std::string line;
    int maxWidth = detailArea.width - 30;
    
    while (words >> word) {
        cv::Size textSize = cv::getTextSize(line + " " + word, UITheme::Fonts::FontFace,
                                           UITheme::Fonts::BodySize, UITheme::Fonts::BodyThickness);
        if (textSize.width > maxWidth && !line.empty()) {
            cv::putText(frame, line, cv::Point(descPos.x, descPos.y + yOffset),
                       UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
                       UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness);
            yOffset += 20;
            line = word;
        } else {
            if (!line.empty()) line += " ";
            line += word;
        }
    }
    if (!line.empty()) {
        cv::putText(frame, line, cv::Point(descPos.x, descPos.y + yOffset),
                   UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
                   UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness);
    }
    
    yOffset += 50;
    
    // Instructions
    cv::Point instrTitlePos(detailArea.x + 15, detailArea.y + yOffset);
    cv::putText(frame, "Instructions:", instrTitlePos, UITheme::Fonts::FontFaceBold,
                UITheme::Fonts::BodySize, UITheme::Colors::TextPrimary,
                UITheme::Fonts::BodyThickness);
    
    yOffset += 25;
    // Similar word wrapping for instructions...
    
    // Drill parameters
    yOffset += 60;
    std::string attemptsText = "Max Attempts: " + std::to_string(drill.maxAttempts);
    cv::Point attemptsPos(detailArea.x + 15, detailArea.y + yOffset);
    cv::putText(frame, attemptsText, attemptsPos, UITheme::Fonts::FontFace,
                UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                UITheme::Fonts::BodyThickness);
    
    yOffset += 25;
    std::ostringstream thresholdStream;
    thresholdStream << "Success Rate: " << std::fixed << std::setprecision(0) 
                   << (drill.successThreshold * 100) << "%";
    cv::Point thresholdPos(detailArea.x + 15, detailArea.y + yOffset);
    cv::putText(frame, thresholdStream.str(), thresholdPos, UITheme::Fonts::FontFace,
                UITheme::Fonts::BodySize, UITheme::Colors::TextSecondary,
                UITheme::Fonts::BodyThickness);
    
    // Start button
    yOffset += 50;
    cv::Rect startButton(detailArea.x + 15, detailArea.y + yOffset, 
                        detailArea.width - 30, BUTTON_HEIGHT);
    UITheme::drawButton(frame, "Start Drill", startButton, false, false, false);
}

void DrillsPage::renderExecutionHUD(cv::Mat& frame) {
    // This would render the drill execution heads-up display
    // with current attempt, accuracy, timer, etc.
}

void DrillsPage::renderAttemptHistory(cv::Mat& frame, const cv::Rect& historyArea) {
    // Render list of attempts with accuracy scores
}

void DrillsPage::renderDrillInstructions(cv::Mat& frame, const cv::Rect& instructionArea) {
    // Render current drill instructions
}

void DrillsPage::renderProgressBar(cv::Mat& frame, const cv::Rect& progressArea) {
    // Render drill progress bar
}

void DrillsPage::renderExecutionControls(cv::Mat& frame, const cv::Rect& controlArea) {
    // Render pause, reset, end drill buttons
}

void DrillsPage::renderResultsSummary(cv::Mat& frame, const cv::Rect& summaryArea) {
    // Render drill completion results
}

void DrillsPage::renderAccuracyChart(cv::Mat& frame, const cv::Rect& chartArea) {
    // Render accuracy over attempts chart
}

void DrillsPage::renderPerformanceMetrics(cv::Mat& frame, const cv::Rect& metricsArea) {
    // Render performance statistics
}

void DrillsPage::renderStatsOverview(cv::Mat& frame, const cv::Rect& overviewArea) {
    // Render overall drill statistics
}

void DrillsPage::renderDrillProgress(cv::Mat& frame, const cv::Rect& progressArea) {
    // Render progress across different drills
}

void DrillsPage::renderImprovementTrends(cv::Mat& frame, const cv::Rect& trendsArea) {
    // Render improvement trends over time
}

void DrillsPage::renderDrillDesigner(cv::Mat& frame, const cv::Rect& designArea) {
    // Render custom drill creation interface
}

void DrillsPage::renderBallPlacement(cv::Mat& frame, const cv::Rect& tableArea) {
    // Render table with ball placement interface
}

void DrillsPage::renderDrillProperties(cv::Mat& frame, const cv::Rect& propertiesArea) {
    // Render drill property editor
}

void DrillsPage::updateFilteredDrills() {
    // Get all drills from drill library
    DrillLibrary library;
    auto allDrills = library.getAllDrills();
    
    filteredDrills_.clear();
    
    for (const auto& drill : allDrills) {
        bool passesFilter = true;
        
        // Category filter
        if (currentFilter_.category != drill.category) {
            // For now, don't filter by category unless specifically selected
            // passesFilter = false;
        }
        
        // Difficulty filter
        if (currentFilter_.difficulty != drill.difficulty) {
            // For now, don't filter by difficulty unless specifically selected
            // passesFilter = false;
        }
        
        // Search filter
        if (!currentFilter_.searchQuery.empty()) {
            std::string query = currentFilter_.searchQuery;
            std::transform(query.begin(), query.end(), query.begin(), ::tolower);
            std::string name = drill.name;
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            std::string desc = drill.description;
            std::transform(desc.begin(), desc.end(), desc.begin(), ::tolower);
            
            if (name.find(query) == std::string::npos && 
                desc.find(query) == std::string::npos) {
                passesFilter = false;
            }
        }
        
        // Custom filter
        if (currentFilter_.showCustomOnly && !drill.isCustom) {
            passesFilter = false;
        }
        
        if (passesFilter) {
            filteredDrills_.push_back(drill);
        }
    }
}

void DrillsPage::startSelectedDrill() {
    if (selectedDrillId_ > 0) {
        if (drillSystem_->startDrill(selectedDrillId_, activePlayerId_)) {
            currentState_ = DrillUIState::DrillExecution;
            executionState_ = DrillExecutionState();
        }
    }
}

void DrillsPage::pauseCurrentDrill() {
    drillSystem_->pauseDrill();
    executionState_.isPaused = true;
}

void DrillsPage::resumeCurrentDrill() {
    drillSystem_->resumeDrill();
    executionState_.isPaused = false;
}

void DrillsPage::endCurrentDrill() {
    drillSystem_->endDrill();
    currentState_ = DrillUIState::DrillResults;
}

void DrillsPage::resetCurrentDrill() {
    drillSystem_->resetDrill();
    executionState_ = DrillExecutionState();
}

bool DrillsPage::isPointInRect(const cv::Point& point, const cv::Rect& rect) const {
    return UITheme::isPointInRect(point, rect);
}

void DrillsPage::createButtons() {
    buttons_.clear();
    
    // Back button
    Button backButton;
    backButton.rect = cv::Rect(10, 10, 80, 30);
    backButton.label = "Back";
    backButton.action = [this]() {
        if (navigationCallback_) {
            navigationCallback_("main_menu");
        }
    };
    buttons_.push_back(backButton);
    
    // Stats button
    Button statsButton;
    statsButton.rect = cv::Rect(100, 10, 100, 30);
    statsButton.label = "Statistics";
    statsButton.action = [this]() {
        currentState_ = DrillUIState::DrillStats;
    };
    buttons_.push_back(statsButton);
    
    // Create Custom button
    Button customButton;
    customButton.rect = cv::Rect(210, 10, 120, 30);
    customButton.label = "Create Custom";
    customButton.action = [this]() {
        currentState_ = DrillUIState::DrillCreator;
    };
    buttons_.push_back(customButton);
}

void DrillsPage::updateButtons() {
    // Update button visibility and states based on current UI state
    for (auto& button : buttons_) {
        button.visible = true;
        button.enabled = true;
    }
}

DrillsPage::Button* DrillsPage::findButtonAt(const cv::Point& point) {
    for (auto& button : buttons_) {
        if (button.visible && button.enabled && UITheme::isPointInRect(point, button.rect)) {
            return &button;
        }
    }
    return nullptr;
}

std::vector<DrillSystem::DrillStats> DrillsPage::getPlayerStats() const {
    return drillSystem_->getPlayerDrillStats(activePlayerId_);
}

void DrillsPage::saveDrillResults() {
    // Save results to database
}

void DrillsPage::loadDrillHistory() {
    // Load drill history from database
}

} // namespace pv