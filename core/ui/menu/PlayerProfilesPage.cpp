#include "PlayerProfilesPage.hpp"
#include <iostream>
#include <algorithm>

namespace pv {

PlayerProfilesPage::PlayerProfilesPage(Database& db)
    : db_(db)
    , currentMode_(Mode::List)
    , windowWidth_(1280)
    , windowHeight_(720)
    , goBack_(false)
    , scrollOffset_(0)
    , activeInputField_(0) {
}

void PlayerProfilesPage::init() {
    currentMode_ = Mode::List;
    goBack_ = false;
    scrollOffset_ = 0;
    activeInputField_ = 0;
    searchQuery_ = "";
    
    // Open database
    if (!db_.isOpen()) {
        db_.open("data/poolvision.db");
    }
    
    loadPlayers();
    updateLayout();
}

void PlayerProfilesPage::setWindowSize(int width, int height) {
    windowWidth_ = width;
    windowHeight_ = height;
    updateLayout();
}

void PlayerProfilesPage::updateLayout() {
    // Top buttons
    addButton_ = cv::Rect(windowWidth_ - 180, 100, 150, 50);
    searchBox_ = cv::Rect(30, 100, 300, 50);
    
    // Bottom buttons
    backButton_ = cv::Rect(30, windowHeight_ - 80, 150, 50);
    saveButton_ = cv::Rect(windowWidth_ - 330, windowHeight_ - 80, 150, 50);
    cancelButton_ = cv::Rect(windowWidth_ - 180, windowHeight_ - 80, 150, 50);
    
    // Form inputs (for add/edit mode)
    int formX = (windowWidth_ - 500) / 2;
    int formY = 180;
    nameInput_ = cv::Rect(formX, formY, 500, 50);
    skillLevelDropdown_ = cv::Rect(formX, formY + 80, 500, 50);
    handednessToggle_ = cv::Rect(formX, formY + 160, 200, 50);
    gameTypeDropdown_ = cv::Rect(formX, formY + 240, 500, 50);
}

void PlayerProfilesPage::loadPlayers() {
    playerItems_.clear();
    
    std::vector<PlayerProfile> players;
    if (searchQuery_.empty()) {
        players = db_.getAllPlayers();
    } else {
        players = db_.searchPlayers(searchQuery_);
    }
    
    int startY = 180;
    int itemHeight = 100;
    int margin = 30;
    
    for (size_t i = 0; i < players.size(); ++i) {
        PlayerListItem item;
        item.profile = players[i];
        
        int y = startY + i * (itemHeight + 10) - scrollOffset_;
        item.rect = cv::Rect(margin, y, windowWidth_ - 2 * margin, itemHeight);
        
        // Action buttons
        int buttonWidth = 80;
        int buttonSpacing = 10;
        int buttonX = item.rect.x + item.rect.width - 3 * (buttonWidth + buttonSpacing);
        int buttonY = item.rect.y + (itemHeight - 40) / 2;
        
        item.viewButton = cv::Rect(buttonX, buttonY, buttonWidth, 40);
        item.editButton = cv::Rect(buttonX + buttonWidth + buttonSpacing, buttonY, buttonWidth, 40);
        item.deleteButton = cv::Rect(buttonX + 2 * (buttonWidth + buttonSpacing), buttonY, buttonWidth, 40);
        
        playerItems_.push_back(item);
    }
}

cv::Mat PlayerProfilesPage::render() {
    cv::Mat img(windowHeight_, windowWidth_, CV_8UC3, UITheme::Colors::DarkBg);
    
    // Draw title
    UITheme::drawTitleBar(img, "Player Profiles", 80);
    
    // Render based on mode
    switch (currentMode_) {
        case Mode::List:
            drawPlayerList(img);
            break;
        case Mode::Add:
        case Mode::Edit:
            drawPlayerForm(img);
            break;
        case Mode::View:
            drawPlayerDetails(img);
            break;
    }
    
    drawActionButtons(img);
    
    return img;
}

void PlayerProfilesPage::drawPlayerList(cv::Mat& img) {
    // Search box
    drawTextInput(img, "Search", searchQuery_, searchBox_, activeInputField_ == 2);
    
    // Add button
    UITheme::drawButton(img, "+ Add Player", addButton_, false, false, false);
    
    // Player list
    for (auto& item : playerItems_) {
        if (item.rect.y + item.rect.height < 80 || item.rect.y > windowHeight_) {
            continue;  // Skip items outside view
        }
        
        // Draw player card
        UITheme::drawCard(img, item.rect, UITheme::Colors::MediumBg, 220);
        
        // Player name
        cv::Point namePos(item.rect.x + 20, item.rect.y + 35);
        UITheme::drawTextWithShadow(img, item.profile.name, namePos,
                                   UITheme::Fonts::FontFaceBold,
                                   UITheme::Fonts::HeadingSize,
                                   UITheme::Colors::TextPrimary,
                                   UITheme::Fonts::HeadingThickness);
        
        // Stats line
        std::string stats = "Games: " + std::to_string(item.profile.gamesPlayed) +
                          " | Win Rate: " + std::to_string(static_cast<int>(item.profile.winRate * 100)) + "%" +
                          " | Skill: " + item.profile.getSkillLevelString();
        cv::Point statsPos(item.rect.x + 20, item.rect.y + 70);
        cv::putText(img, stats, statsPos, UITheme::Fonts::FontFace,
                   UITheme::Fonts::SmallSize, UITheme::Colors::TextSecondary,
                   UITheme::Fonts::BodyThickness);
        
        // Action buttons
        UITheme::drawButton(img, "View", item.viewButton, false, false, false);
        UITheme::drawButton(img, "Edit", item.editButton, false, false, false);
        UITheme::drawButton(img, "Delete", item.deleteButton, false, false, false);
    }
    
    // Show count
    std::string countText = std::to_string(playerItems_.size()) + " players";
    cv::Size textSize = UITheme::getTextSize(countText, UITheme::Fonts::FontFace,
                                             UITheme::Fonts::SmallSize,
                                             UITheme::Fonts::BodyThickness);
    cv::Point countPos(windowWidth_ - textSize.width - 30, 135);
    cv::putText(img, countText, countPos, UITheme::Fonts::FontFace,
               UITheme::Fonts::SmallSize, UITheme::Colors::TextDisabled,
               UITheme::Fonts::BodyThickness);
}

void PlayerProfilesPage::drawPlayerForm(cv::Mat& img) {
    std::string title = (currentMode_ == Mode::Add) ? "Add New Player" : "Edit Player";
    
    // Form title
    cv::Size titleSize = UITheme::getTextSize(title, UITheme::Fonts::FontFaceBold,
                                              UITheme::Fonts::HeadingSize,
                                              UITheme::Fonts::HeadingThickness);
    cv::Point titlePos((windowWidth_ - titleSize.width) / 2, 140);
    UITheme::drawTextWithShadow(img, title, titlePos, UITheme::Fonts::FontFaceBold,
                                UITheme::Fonts::HeadingSize,
                                UITheme::Colors::TextPrimary,
                                UITheme::Fonts::HeadingThickness);
    
    // Name input
    drawTextInput(img, "Player Name", currentPlayer_.name, nameInput_, activeInputField_ == 1);
    
    // Skill level
    cv::Point skillLabel(skillLevelDropdown_.x, skillLevelDropdown_.y - 10);
    cv::putText(img, "Skill Level", skillLabel, UITheme::Fonts::FontFace,
               UITheme::Fonts::SmallSize, UITheme::Colors::TextSecondary,
               UITheme::Fonts::BodyThickness);
    drawSkillLevelSelector(img, skillLevelDropdown_);
    
    // Handedness
    cv::Point handLabel(handednessToggle_.x, handednessToggle_.y - 10);
    cv::putText(img, "Handedness", handLabel, UITheme::Fonts::FontFace,
               UITheme::Fonts::SmallSize, UITheme::Colors::TextSecondary,
               UITheme::Fonts::BodyThickness);
    drawHandednessSelector(img, handednessToggle_);
    
    // Preferred game type
    cv::Point gameLabel(gameTypeDropdown_.x, gameTypeDropdown_.y - 10);
    cv::putText(img, "Preferred Game Type", gameLabel, UITheme::Fonts::FontFace,
               UITheme::Fonts::SmallSize, UITheme::Colors::TextSecondary,
               UITheme::Fonts::BodyThickness);
    UITheme::drawDropdown(img, currentPlayer_.preferredGameType, gameTypeDropdown_);
}

void PlayerProfilesPage::drawPlayerDetails(cv::Mat& img) {
    // Player name as title
    cv::Size titleSize = UITheme::getTextSize(currentPlayer_.name, UITheme::Fonts::FontFaceBold,
                                              UITheme::Fonts::TitleSize,
                                              UITheme::Fonts::TitleThickness);
    cv::Point titlePos((windowWidth_ - titleSize.width) / 2, 140);
    UITheme::drawTextWithShadow(img, currentPlayer_.name, titlePos, UITheme::Fonts::FontFaceBold,
                                UITheme::Fonts::TitleSize,
                                UITheme::Colors::NeonCyan,
                                UITheme::Fonts::TitleThickness);
    
    // Stats cards
    int cardWidth = 250;
    int cardHeight = 120;
    int cardSpacing = 30;
    int startX = (windowWidth_ - 3 * cardWidth - 2 * cardSpacing) / 2;
    int startY = 200;
    
    // Games played card
    cv::Rect gamesCard(startX, startY, cardWidth, cardHeight);
    UITheme::drawCard(img, gamesCard);
    std::string gamesText = std::to_string(currentPlayer_.gamesPlayed);
    cv::Size gamesSize = UITheme::getTextSize(gamesText, UITheme::Fonts::FontFaceBold,
                                              UITheme::Fonts::TitleSize * 1.2,
                                              UITheme::Fonts::TitleThickness);
    cv::Point gamesPos(gamesCard.x + (gamesCard.width - gamesSize.width) / 2,
                      gamesCard.y + 60);
    UITheme::drawTextWithShadow(img, gamesText, gamesPos, UITheme::Fonts::FontFaceBold,
                                UITheme::Fonts::TitleSize * 1.2,
                                UITheme::Colors::NeonYellow,
                                UITheme::Fonts::TitleThickness);
    cv::Point gamesLabel(gamesCard.x + 20, gamesCard.y + 100);
    cv::putText(img, "Games Played", gamesLabel, UITheme::Fonts::FontFace,
               UITheme::Fonts::SmallSize, UITheme::Colors::TextSecondary,
               UITheme::Fonts::BodyThickness);
    
    // Win rate card
    cv::Rect winCard(startX + cardWidth + cardSpacing, startY, cardWidth, cardHeight);
    UITheme::drawCard(img, winCard);
    std::string winText = std::to_string(static_cast<int>(currentPlayer_.winRate * 100)) + "%";
    cv::Size winSize = UITheme::getTextSize(winText, UITheme::Fonts::FontFaceBold,
                                            UITheme::Fonts::TitleSize * 1.2,
                                            UITheme::Fonts::TitleThickness);
    cv::Point winPos(winCard.x + (winCard.width - winSize.width) / 2,
                    winCard.y + 60);
    UITheme::drawTextWithShadow(img, winText, winPos, UITheme::Fonts::FontFaceBold,
                                UITheme::Fonts::TitleSize * 1.2,
                                UITheme::Colors::NeonGreen,
                                UITheme::Fonts::TitleThickness);
    cv::Point winLabel(winCard.x + 20, winCard.y + 100);
    cv::putText(img, "Win Rate", winLabel, UITheme::Fonts::FontFace,
               UITheme::Fonts::SmallSize, UITheme::Colors::TextSecondary,
               UITheme::Fonts::BodyThickness);
    
    // Shot success card
    cv::Rect shotCard(startX + 2 * (cardWidth + cardSpacing), startY, cardWidth, cardHeight);
    UITheme::drawCard(img, shotCard);
    std::string shotText = std::to_string(static_cast<int>(currentPlayer_.shotSuccessRate * 100)) + "%";
    cv::Size shotSize = UITheme::getTextSize(shotText, UITheme::Fonts::FontFaceBold,
                                             UITheme::Fonts::TitleSize * 1.2,
                                             UITheme::Fonts::TitleThickness);
    cv::Point shotPos(shotCard.x + (shotCard.width - shotSize.width) / 2,
                     shotCard.y + 60);
    UITheme::drawTextWithShadow(img, shotText, shotPos, UITheme::Fonts::FontFaceBold,
                                UITheme::Fonts::TitleSize * 1.2,
                                UITheme::Colors::NeonCyan,
                                UITheme::Fonts::TitleThickness);
    cv::Point shotLabel(shotCard.x + 20, shotCard.y + 100);
    cv::putText(img, "Shot Success", shotLabel, UITheme::Fonts::FontFace,
               UITheme::Fonts::SmallSize, UITheme::Colors::TextSecondary,
               UITheme::Fonts::BodyThickness);
    
    // Profile info
    int infoY = 360;
    int lineHeight = 40;
    
    std::string skillText = "Skill Level: " + currentPlayer_.getSkillLevelString();
    cv::Point skillPos((windowWidth_ - 400) / 2, infoY);
    cv::putText(img, skillText, skillPos, UITheme::Fonts::FontFace,
               UITheme::Fonts::BodySize, UITheme::Colors::TextPrimary,
               UITheme::Fonts::BodyThickness);
    
    std::string handText = "Handedness: " + currentPlayer_.getHandednessString();
    cv::Point handPos((windowWidth_ - 400) / 2, infoY + lineHeight);
    cv::putText(img, handText, handPos, UITheme::Fonts::FontFace,
               UITheme::Fonts::BodySize, UITheme::Colors::TextPrimary,
               UITheme::Fonts::BodyThickness);
    
    std::string gameText = "Preferred Game: " + currentPlayer_.preferredGameType;
    cv::Point gamePos((windowWidth_ - 400) / 2, infoY + 2 * lineHeight);
    cv::putText(img, gameText, gamePos, UITheme::Fonts::FontFace,
               UITheme::Fonts::BodySize, UITheme::Colors::TextPrimary,
               UITheme::Fonts::BodyThickness);
}

void PlayerProfilesPage::drawActionButtons(cv::Mat& img) {
    switch (currentMode_) {
        case Mode::List:
            UITheme::drawButton(img, "< Back", backButton_);
            break;
        case Mode::Add:
        case Mode::Edit:
            UITheme::drawButton(img, "Cancel", cancelButton_);
            UITheme::drawButton(img, "Save", saveButton_, false, true, !currentPlayer_.isValid());
            break;
        case Mode::View:
            UITheme::drawButton(img, "< Back to List", backButton_);
            break;
    }
}

void PlayerProfilesPage::drawTextInput(cv::Mat& img, const std::string& label,
                                      const std::string& value, const cv::Rect& rect, bool active) {
    // Label
    cv::Point labelPos(rect.x, rect.y - 10);
    cv::putText(img, label, labelPos, UITheme::Fonts::FontFace,
               UITheme::Fonts::SmallSize, UITheme::Colors::TextSecondary,
               UITheme::Fonts::BodyThickness);
    
    // Input box
    cv::Scalar borderColor = active ? UITheme::Colors::NeonCyan : UITheme::Colors::BorderColor;
    UITheme::drawRoundedRect(img, rect, 5, UITheme::Colors::MediumBg, -1);
    UITheme::drawRoundedRect(img, rect, 5, borderColor, 2);
    
    // Text
    cv::Point textPos(rect.x + 15, rect.y + rect.height / 2 + 5);
    std::string displayText = value.empty() ? "Enter " + label : value;
    cv::Scalar textColor = value.empty() ? UITheme::Colors::TextDisabled : UITheme::Colors::TextPrimary;
    cv::putText(img, displayText, textPos, UITheme::Fonts::FontFace,
               UITheme::Fonts::BodySize, textColor,
               UITheme::Fonts::BodyThickness);
    
    // Cursor for active input
    if (active) {
        int cursorX = rect.x + 15 + static_cast<int>(value.length() * 12);
        cv::line(img, cv::Point(cursorX, rect.y + 15),
                cv::Point(cursorX, rect.y + rect.height - 15),
                UITheme::Colors::NeonCyan, 2);
    }
}

void PlayerProfilesPage::drawSkillLevelSelector(cv::Mat& img, const cv::Rect& rect) {
    UITheme::drawDropdown(img, currentPlayer_.getSkillLevelString(), rect);
}

void PlayerProfilesPage::drawHandednessSelector(cv::Mat& img, const cv::Rect& rect) {
    // Draw three buttons for Right, Left, Ambidextrous
    int buttonWidth = 120;
    int spacing = 10;
    
    std::vector<std::string> options = {"Right", "Left", "Ambidex"};
    std::vector<Handedness> values = {Handedness::Right, Handedness::Left, Handedness::Ambidextrous};
    
    for (size_t i = 0; i < options.size(); ++i) {
        cv::Rect buttonRect(rect.x + i * (buttonWidth + spacing), rect.y, buttonWidth, rect.height);
        bool isActive = (currentPlayer_.handedness == values[i]);
        UITheme::drawButton(img, options[i], buttonRect, false, isActive, false);
    }
}

// Event handlers

void PlayerProfilesPage::onMouse(int event, int x, int y, int flags) {
    mousePos_ = cv::Point(x, y);
    
    if (event != cv::EVENT_LBUTTONDOWN) return;
    
    switch (currentMode_) {
        case Mode::List:
            handleListClick(x, y);
            break;
        case Mode::Add:
        case Mode::Edit:
            handleFormClick(x, y);
            break;
        case Mode::View:
            handleDetailsClick(x, y);
            break;
    }
}

void PlayerProfilesPage::handleListClick(int x, int y) {
    // Check add button
    if (UITheme::isPointInRect(mousePos_, addButton_)) {
        startAddPlayer();
        return;
    }
    
    // Check back button
    if (UITheme::isPointInRect(mousePos_, backButton_)) {
        goBack_ = true;
        return;
    }
    
    // Check search box
    if (UITheme::isPointInRect(mousePos_, searchBox_)) {
        activeInputField_ = 2;
        return;
    }
    
    // Check player items
    for (auto& item : playerItems_) {
        if (UITheme::isPointInRect(mousePos_, item.viewButton)) {
            startViewPlayer(item.profile.id);
            return;
        }
        if (UITheme::isPointInRect(mousePos_, item.editButton)) {
            startEditPlayer(item.profile.id);
            return;
        }
        if (UITheme::isPointInRect(mousePos_, item.deleteButton)) {
            deletePlayer(item.profile.id);
            return;
        }
    }
}

void PlayerProfilesPage::handleFormClick(int x, int y) {
    // Check save button
    if (UITheme::isPointInRect(mousePos_, saveButton_) && currentPlayer_.isValid()) {
        savePlayer();
        return;
    }
    
    // Check cancel button
    if (UITheme::isPointInRect(mousePos_, cancelButton_)) {
        cancelEdit();
        return;
    }
    
    // Check name input
    if (UITheme::isPointInRect(mousePos_, nameInput_)) {
        activeInputField_ = 1;
        return;
    }
    
    // Check skill level dropdown
    if (UITheme::isPointInRect(mousePos_, skillLevelDropdown_)) {
        // Cycle through skill levels
        int level = static_cast<int>(currentPlayer_.skillLevel);
        level = (level % 5) + 1;
        currentPlayer_.skillLevel = static_cast<SkillLevel>(level);
        return;
    }
    
    // Check handedness buttons
    if (UITheme::isPointInRect(mousePos_, handednessToggle_)) {
        int buttonWidth = 120;
        int spacing = 10;
        int relX = x - handednessToggle_.x;
        int buttonIndex = relX / (buttonWidth + spacing);
        
        if (buttonIndex >= 0 && buttonIndex < 3) {
            currentPlayer_.handedness = static_cast<Handedness>(buttonIndex);
        }
        return;
    }
}

void PlayerProfilesPage::handleDetailsClick(int x, int y) {
    if (UITheme::isPointInRect(mousePos_, backButton_)) {
        currentMode_ = Mode::List;
        loadPlayers();
    }
}

// Actions

void PlayerProfilesPage::startAddPlayer() {
    currentPlayer_ = PlayerProfile();
    currentPlayer_.name = "";
    currentPlayer_.skillLevel = SkillLevel::Intermediate;
    currentPlayer_.handedness = Handedness::Right;
    currentPlayer_.preferredGameType = "8-Ball";
    currentMode_ = Mode::Add;
    activeInputField_ = 1;
}

void PlayerProfilesPage::startEditPlayer(int playerId) {
    currentPlayer_ = db_.getPlayer(playerId);
    currentMode_ = Mode::Edit;
    activeInputField_ = 0;
}

void PlayerProfilesPage::startViewPlayer(int playerId) {
    currentPlayer_ = db_.getPlayer(playerId);
    db_.updatePlayerStats(playerId);
    currentPlayer_ = db_.getPlayer(playerId);  // Reload with updated stats
    currentMode_ = Mode::View;
}

void PlayerProfilesPage::savePlayer() {
    if (!currentPlayer_.isValid()) return;
    
    if (currentMode_ == Mode::Add) {
        int id = db_.createPlayer(currentPlayer_);
        if (id > 0) {
            std::cout << "✓ Player created: " << currentPlayer_.name << std::endl;
        }
    } else if (currentMode_ == Mode::Edit) {
        if (db_.updatePlayer(currentPlayer_)) {
            std::cout << "✓ Player updated: " << currentPlayer_.name << std::endl;
        }
    }
    
    currentMode_ = Mode::List;
    loadPlayers();
}

void PlayerProfilesPage::deletePlayer(int playerId) {
    if (db_.deletePlayer(playerId)) {
        std::cout << "✓ Player deleted" << std::endl;
        loadPlayers();
    }
}

void PlayerProfilesPage::cancelEdit() {
    currentMode_ = Mode::List;
    activeInputField_ = 0;
    loadPlayers();
}

void PlayerProfilesPage::onKey(int key) {
    // ESC to go back
    if (key == 27) {
        if (currentMode_ == Mode::List) {
            goBack_ = true;
        } else {
            cancelEdit();
        }
        return;
    }
    
    // Handle text input
    if (activeInputField_ == 1 && currentMode_ != Mode::View) {
        if (key == 8 && !currentPlayer_.name.empty()) {  // Backspace
            currentPlayer_.name.pop_back();
        } else if (key >= 32 && key <= 126 && currentPlayer_.name.length() < 30) {
            currentPlayer_.name += static_cast<char>(key);
        }
    } else if (activeInputField_ == 2 && currentMode_ == Mode::List) {
        if (key == 8 && !searchQuery_.empty()) {  // Backspace
            searchQuery_.pop_back();
            loadPlayers();
        } else if (key == 13) {  // Enter
            loadPlayers();
        } else if (key >= 32 && key <= 126 && searchQuery_.length() < 30) {
            searchQuery_ += static_cast<char>(key);
            loadPlayers();
        }
    }
}

} // namespace pv
