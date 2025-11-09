#include "PlayerProfilesPage.hpp"
#include "../ResponsiveLayout.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>

namespace pv {

PlayerProfilesPage::PlayerProfilesPage(Database& db)
    : db_(db)
    , currentMode_(Mode::List)
    , windowWidth_(1280)
    , windowHeight_(720)
    , goBack_(false)
    , scrollOffset_(0)
    , activeInputField_(0)
    , rootContainer_(nullptr)
    , animationTime_(0.0f) {
}

void PlayerProfilesPage::init() {
    currentMode_ = Mode::List;
    goBack_ = false;
    scrollOffset_ = 0;
    activeInputField_ = 0;
    searchQuery_ = "";
    
    // Initialize UITheme for responsive scaling
    UITheme::init(windowWidth_, windowHeight_);
    
    // Open database
    if (!db_.isOpen()) {
        db_.open("data/poolvision.db");
    }
    
    loadPlayers();
    createResponsiveLayout();
}

void PlayerProfilesPage::setWindowSize(int width, int height) {
    windowWidth_ = width;
    windowHeight_ = height;
    
    // Update UITheme scaling
    UITheme::setWindowSize(width, height);
    
    createResponsiveLayout();
    loadPlayers(); // Recalculate card positions
}

void PlayerProfilesPage::createResponsiveLayout() {
    // Create root container with full window size
    cv::Rect windowRect(0, 0, windowWidth_, windowHeight_);
    rootContainer_ = std::make_unique<ResponsiveLayout::Container>(
        ResponsiveLayout::Direction::Column, windowRect);
    
    rootContainer_->setJustifyContent(ResponsiveLayout::Justify::SpaceBetween);
    rootContainer_->setAlignItems(ResponsiveLayout::Alignment::Center);
    rootContainer_->setPadding(UITheme::getResponsiveSpacing(20));
    
    // Header area (title + controls) - 20% of height
    headerRect_ = UITheme::getResponsiveRect(0, 0, 100, 20, windowRect);
    
    // Content area (player grid/form) - 70% of height
    contentRect_ = UITheme::getResponsiveRect(0, 20, 100, 70, windowRect);
    
    // Button area - 10% of height
    buttonRect_ = UITheme::getResponsiveRect(0, 90, 100, 10, windowRect);
    
    updateControlLayout();
}

void PlayerProfilesPage::updateControlLayout() {
    int padding = UITheme::getResponsiveSpacing(30);
    
    // Top control bar layout
    cv::Size buttonSize = UITheme::getResponsiveSize(180, 50);
    cv::Size searchSize = UITheme::getResponsiveSize(350, 50);
    
    // Search box (left side)
    searchBox_ = cv::Rect(headerRect_.x + padding,
                         headerRect_.y + headerRect_.height - buttonSize.height - padding,
                         searchSize.width, searchSize.height);
    
    // Add button (right side)
    addButton_ = cv::Rect(headerRect_.x + headerRect_.width - buttonSize.width - padding,
                         headerRect_.y + headerRect_.height - buttonSize.height - padding,
                         buttonSize.width, buttonSize.height);
    
    // Bottom button layout
    buttonSize = UITheme::getResponsiveSize(150, 50);
    int buttonSpacing = UITheme::getResponsiveSpacing(20);
    
    backButton_ = cv::Rect(buttonRect_.x + padding,
                          buttonRect_.y + (buttonRect_.height - buttonSize.height) / 2,
                          buttonSize.width, buttonSize.height);
    
    saveButton_ = cv::Rect(buttonRect_.x + buttonRect_.width - buttonSize.width * 2 - buttonSpacing - padding,
                          buttonRect_.y + (buttonRect_.height - buttonSize.height) / 2,
                          buttonSize.width, buttonSize.height);
    
    cancelButton_ = cv::Rect(buttonRect_.x + buttonRect_.width - buttonSize.width - padding,
                            buttonRect_.y + (buttonRect_.height - buttonSize.height) / 2,
                            buttonSize.width, buttonSize.height);
    
    // Form layout (responsive)
    updateFormLayout();
}

void PlayerProfilesPage::updateFormLayout() {
    int formPadding = UITheme::getResponsiveSpacing(50);
    int formWidth = std::min(static_cast<int>(contentRect_.width * 0.8f), 
                            UITheme::getResponsiveSize(600, 50).width);
    int formX = contentRect_.x + (contentRect_.width - formWidth) / 2;
    int rowHeight = UITheme::getResponsiveSpacing(80);
    
    int startY = contentRect_.y + UITheme::getResponsiveSpacing(40);
    
    nameInput_ = cv::Rect(formX, startY, formWidth, UITheme::getResponsiveSpacing(50));
    startY += rowHeight;
    
    skillLevelDropdown_ = cv::Rect(formX, startY, formWidth, UITheme::getResponsiveSpacing(50));
    startY += rowHeight;
    
    handednessToggle_ = cv::Rect(formX, startY, UITheme::getResponsiveSpacing(200), UITheme::getResponsiveSpacing(50));
    startY += rowHeight;
    
    gameTypeDropdown_ = cv::Rect(formX, startY, formWidth, UITheme::getResponsiveSpacing(50));
}

// Legacy method for compatibility
void PlayerProfilesPage::updateLayout() {
    createResponsiveLayout();
}

void PlayerProfilesPage::loadPlayers() {
    playerItems_.clear();
    
    std::vector<PlayerProfile> players;
    if (searchQuery_.empty()) {
        players = db_.getAllPlayers();
    } else {
        players = db_.searchPlayers(searchQuery_);
    }
    
    // Calculate responsive card grid layout
    int cardPadding = UITheme::getResponsiveSpacing(20);
    cv::Size cardSize = calculateResponsiveCardSize();
    
    // Calculate grid dimensions
    int availableWidth = contentRect_.width - cardPadding * 2;
    int cardsPerRow = std::max(1, availableWidth / (cardSize.width + UITheme::getResponsiveSpacing(20)));
    int actualCardSpacing = (availableWidth - cardsPerRow * cardSize.width) / std::max(1, cardsPerRow - 1);
    if (cardsPerRow == 1) actualCardSpacing = 0;
    
    int rowSpacing = UITheme::getResponsiveSpacing(25);
    int startY = contentRect_.y + cardPadding;
    
    for (size_t i = 0; i < players.size(); ++i) {
        PlayerListItem item;
        item.profile = players[i];
        
        int row = static_cast<int>(i) / cardsPerRow;
        int col = static_cast<int>(i) % cardsPerRow;
        
        int x = contentRect_.x + cardPadding + col * (cardSize.width + actualCardSpacing);
        int y = startY + row * (cardSize.height + rowSpacing) - scrollOffset_;
        
        item.rect = cv::Rect(x, y, cardSize.width, cardSize.height);
        
        // Calculate button layout within card
        calculateCardButtons(item, cardSize);
        
        playerItems_.push_back(item);
    }
}

cv::Size PlayerProfilesPage::calculateResponsiveCardSize() {
    // Responsive card sizing based on screen size
    int baseWidth = UITheme::getResponsiveSize(300, 180).width;
    int baseHeight = UITheme::getResponsiveSize(300, 180).height;
    
    // Ensure minimum and maximum sizes
    baseWidth = std::clamp(baseWidth, 250, 400);
    baseHeight = std::clamp(baseHeight, 150, 250);
    
    return cv::Size(baseWidth, baseHeight);
}

void PlayerProfilesPage::calculateCardButtons(PlayerListItem& item, const cv::Size& cardSize) {
    int buttonWidth = UITheme::getResponsiveSpacing(70);
    int buttonHeight = UITheme::getResponsiveSpacing(35);
    int buttonSpacing = UITheme::getResponsiveSpacing(8);
    
    // Position buttons at bottom of card
    int totalButtonWidth = 3 * buttonWidth + 2 * buttonSpacing;
    int buttonStartX = item.rect.x + (cardSize.width - totalButtonWidth) / 2;
    int buttonY = item.rect.y + cardSize.height - buttonHeight - UITheme::getResponsiveSpacing(15);
    
    item.viewButton = cv::Rect(buttonStartX, buttonY, buttonWidth, buttonHeight);
    item.editButton = cv::Rect(buttonStartX + buttonWidth + buttonSpacing, buttonY, buttonWidth, buttonHeight);
    item.deleteButton = cv::Rect(buttonStartX + 2 * (buttonWidth + buttonSpacing), buttonY, buttonWidth, buttonHeight);
}

cv::Mat PlayerProfilesPage::render() {
    cv::Mat img(windowHeight_, windowWidth_, CV_8UC3, UITheme::Colors::DarkBg);
    
    // Update animation time
    static auto startTime = std::chrono::steady_clock::now();
    auto currentTime = std::chrono::steady_clock::now();
    animationTime_ = std::chrono::duration<float>(currentTime - startTime).count();
    
    // Draw enhanced background
    drawBackground(img);
    
    // Draw modern title
    drawTitle(img);
    
    // Render based on mode with enhanced visuals
    switch (currentMode_) {
        case Mode::List:
            drawPlayerGrid(img);
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

void PlayerProfilesPage::drawBackground(cv::Mat& img) {
    // Fill with dark background
    img.setTo(UITheme::Colors::DarkBg);
    
    // Add subtle animated background
    UITheme::drawAnimatedBackground(img, animationTime_, 0.15f);
    
    // Add glass-morphism overlay
    cv::Rect overlayRect(0, 0, windowWidth_, windowHeight_);
    UITheme::applyGlassMorphism(img, overlayRect, 8, 0.03f, UITheme::Colors::MediumBg);
}

void PlayerProfilesPage::drawTitle(cv::Mat& img) {
    std::string title = "PLAYER PROFILES";
    double titleSize = UITheme::getResponsiveFontSize(UITheme::Fonts::TitleSize * 1.2);
    
    cv::Size titleTextSize = UITheme::getTextSize(title, UITheme::Fonts::FontFaceBold,
                                                 titleSize, UITheme::Fonts::TitleThickness);
    
    // Create glass card for title
    int titlePadding = UITheme::getResponsiveSpacing(25);
    cv::Rect titleCard(
        (windowWidth_ - titleTextSize.width - titlePadding * 2) / 2,
        headerRect_.y + UITheme::getResponsiveSpacing(10),
        titleTextSize.width + titlePadding * 2,
        titleTextSize.height + titlePadding
    );
    
    // Draw glass card
    UITheme::drawGlassCard(img, titleCard, 12.0f, 0.1f, UITheme::Colors::MediumBg);
    UITheme::drawRoundedRect(img, titleCard, UITheme::getResponsiveSpacing(12),
                           cv::Scalar(UITheme::Colors::MediumBg[0], 
                                     UITheme::Colors::MediumBg[1], 
                                     UITheme::Colors::MediumBg[2], 100), -1, true);
    
    // Draw border
    UITheme::drawRoundedRect(img, titleCard, UITheme::getResponsiveSpacing(12),
                           UITheme::Colors::NeonCyan, 2, true);
    
    // Draw title text
    cv::Point titlePos(titleCard.x + titlePadding,
                      titleCard.y + titlePadding + titleTextSize.height - 5);
    
    UITheme::drawTextWithShadow(img, title, titlePos, UITheme::Fonts::FontFaceBold,
                                titleSize, UITheme::Colors::NeonCyan,
                                UITheme::Fonts::TitleThickness, 
                                UITheme::getResponsiveSpacing(3), true);
}

void PlayerProfilesPage::drawPlayerGrid(cv::Mat& img) {
    // Draw search controls with glass effect
    drawSearchControls(img);
    
    // Draw responsive player card grid
    for (auto& item : playerItems_) {
        if (item.rect.y + item.rect.height < contentRect_.y || 
            item.rect.y > contentRect_.y + contentRect_.height) {
            continue;  // Skip items outside view
        }
        
        drawPlayerCard(img, item);
    }
    
    // Draw stats footer
    drawStatsFooter(img);
}

void PlayerProfilesPage::drawSearchControls(cv::Mat& img) {
    // Search box with glass background
    cv::Rect searchContainer = searchBox_;
    searchContainer.x -= UITheme::getResponsiveSpacing(10);
    searchContainer.y -= UITheme::getResponsiveSpacing(10);
    searchContainer.width += UITheme::getResponsiveSpacing(20);
    searchContainer.height += UITheme::getResponsiveSpacing(20);
    
    UITheme::drawGlassCard(img, searchContainer, 10.0f, 0.1f, UITheme::Colors::MediumBg);
    
    // Draw search input
    UITheme::ComponentState searchState = (activeInputField_ == 2) ? 
        UITheme::ComponentState::Focused : UITheme::ComponentState::Normal;
    
    drawTextInput(img, "🔍 Search players...", searchQuery_, searchBox_, searchState);
    
    // Add button with enhanced styling
    UITheme::ComponentState addState = UITheme::ComponentState::Normal;
    if (UITheme::isPointInRect(mousePos_, addButton_)) {
        addState = UITheme::ComponentState::Hover;
    }
    
    UITheme::drawIconButton(img, "➕", "Add Player", addButton_, addState);
}

void PlayerProfilesPage::drawPlayerCard(cv::Mat& img, PlayerListItem& item) {
    // Determine card state
    UITheme::ComponentState cardState = UITheme::ComponentState::Normal;
    if (item.isHovered) {
        cardState = UITheme::ComponentState::Hover;
    }
    
    // Draw enhanced card with glass-morphism
    int elevation = (cardState == UITheme::ComponentState::Hover) ? 8 : 4;
    UITheme::drawCard(img, item.rect, cardState, UITheme::Colors::MediumBg, elevation);
    
    // Add glass overlay
    UITheme::drawGlassCard(img, item.rect, 8.0f, 0.05f, UITheme::Colors::LightBg);
    
    // Player avatar/icon area
    int avatarSize = UITheme::getResponsiveSpacing(60);
    cv::Rect avatarRect(item.rect.x + UITheme::getResponsiveSpacing(20),
                       item.rect.y + UITheme::getResponsiveSpacing(15),
                       avatarSize, avatarSize);
    
    // Draw avatar placeholder
    UITheme::drawRoundedRect(img, avatarRect, avatarSize / 2, UITheme::Colors::NeonCyan, -1, true);
    
    // Player initial in avatar
    std::string initial = item.profile.name.empty() ? "?" : 
                         std::string(1, std::toupper(item.profile.name[0]));
    double avatarFontSize = UITheme::getResponsiveFontSize(UITheme::Fonts::HeadingSize);
    cv::Size initialSize = UITheme::getTextSize(initial, UITheme::Fonts::FontFaceBold,
                                              avatarFontSize, UITheme::Fonts::HeadingThickness);
    cv::Point initialPos(avatarRect.x + (avatarSize - initialSize.width) / 2,
                        avatarRect.y + (avatarSize + initialSize.height) / 2);
    
    UITheme::drawText(img, initial, initialPos, avatarFontSize,
                     UITheme::Colors::DarkBg, UITheme::Fonts::HeadingThickness, true);
    
    // Player name
    int textStartX = avatarRect.x + avatarSize + UITheme::getResponsiveSpacing(15);
    cv::Point namePos(textStartX, item.rect.y + UITheme::getResponsiveSpacing(35));
    double nameFontSize = UITheme::getResponsiveFontSize(UITheme::Fonts::HeadingSize);
    
    UITheme::drawText(img, item.profile.name, namePos, nameFontSize,
                     UITheme::Colors::TextPrimary, UITheme::Fonts::HeadingThickness, true);
    
    // Stats with icons
    drawPlayerStats(img, item, textStartX, item.rect.y + UITheme::getResponsiveSpacing(65));
    
    // Action buttons with enhanced styling
    drawCardButtons(img, item);
    
    // Add hover glow effect
    if (item.isHovered) {
        cv::Rect glowRect = item.rect;
        int glowExpansion = UITheme::getResponsiveSpacing(6);
        glowRect.x -= glowExpansion;
        glowRect.y -= glowExpansion;
        glowRect.width += 2 * glowExpansion;
        glowRect.height += 2 * glowExpansion;
        
        cv::Scalar glowColor = UITheme::Colors::NeonCyan;
        glowColor[3] = 80; // Semi-transparent
        
        UITheme::drawRoundedRect(img, glowRect, 
                               UITheme::getResponsiveSpacing(UITheme::Layout::BorderRadius + 3),
                               glowColor, 2, true);
    }
}

void PlayerProfilesPage::drawPlayerStats(cv::Mat& img, const PlayerListItem& item, int x, int y) {
    double statsFontSize = UITheme::getResponsiveFontSize(UITheme::Fonts::SmallSize);
    int lineHeight = UITheme::getResponsiveSpacing(18);
    
    // Games played
    std::string gamesText = "🎮 " + std::to_string(item.profile.gamesPlayed) + " games";
    UITheme::drawText(img, gamesText, cv::Point(x, y), statsFontSize,
                     UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness, true);
    
    // Win rate
    y += lineHeight;
    std::string winRateText = "🏆 " + std::to_string(static_cast<int>(item.profile.winRate * 100)) + "% wins";
    UITheme::drawText(img, winRateText, cv::Point(x, y), statsFontSize,
                     UITheme::Colors::NeonGreen, UITheme::Fonts::BodyThickness, true);
    
    // Skill level
    y += lineHeight;
    std::string skillText = "⭐ " + item.profile.getSkillLevelString();
    UITheme::drawText(img, skillText, cv::Point(x, y), statsFontSize,
                     UITheme::Colors::NeonYellow, UITheme::Fonts::BodyThickness, true);
}

void PlayerProfilesPage::drawCardButtons(cv::Mat& img, const PlayerListItem& item) {
    // Button states
    UITheme::ComponentState viewState = UITheme::ComponentState::Normal;
    UITheme::ComponentState editState = UITheme::ComponentState::Normal;
    UITheme::ComponentState deleteState = UITheme::ComponentState::Normal;
    
    if (UITheme::isPointInRect(mousePos_, item.viewButton)) viewState = UITheme::ComponentState::Hover;
    if (UITheme::isPointInRect(mousePos_, item.editButton)) editState = UITheme::ComponentState::Hover;
    if (UITheme::isPointInRect(mousePos_, item.deleteButton)) deleteState = UITheme::ComponentState::Hover;
    
    // Draw buttons with modern styling
    UITheme::drawButton(img, "View", item.viewButton, viewState);
    UITheme::drawButton(img, "Edit", item.editButton, editState);
    
    // Delete button with warning color
    cv::Scalar deleteColor = UITheme::Colors::NeonRed;
    UITheme::drawRoundedRect(img, item.deleteButton, UITheme::getResponsiveSpacing(UITheme::Layout::BorderRadius),
                           deleteColor, -1, true);
    
    double buttonFontSize = UITheme::getResponsiveFontSize(UITheme::Fonts::SmallSize);
    cv::Size deleteTextSize = UITheme::getTextSize("Delete", UITheme::Fonts::FontFace,
                                                  buttonFontSize, UITheme::Fonts::BodyThickness);
    cv::Point deleteTextPos(item.deleteButton.x + (item.deleteButton.width - deleteTextSize.width) / 2,
                           item.deleteButton.y + (item.deleteButton.height + deleteTextSize.height) / 2);
    
    UITheme::drawText(img, "Delete", deleteTextPos, buttonFontSize,
                     UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness, true);
}

void PlayerProfilesPage::drawStatsFooter(cv::Mat& img) {
    // Footer with player count and stats
    std::string countText = std::to_string(playerItems_.size()) + " players total";
    double footerFontSize = UITheme::getResponsiveFontSize(UITheme::Fonts::SmallSize);
    cv::Size textSize = UITheme::getTextSize(countText, UITheme::Fonts::FontFace,
                                           footerFontSize, UITheme::Fonts::BodyThickness);
    
    cv::Point countPos(contentRect_.x + UITheme::getResponsiveSpacing(20),
                      contentRect_.y + contentRect_.height - UITheme::getResponsiveSpacing(10));
    
    UITheme::drawText(img, countText, countPos, footerFontSize,
                     UITheme::Colors::TextDisabled, UITheme::Fonts::BodyThickness, true);
}
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
