#include "ShotLibrary.hpp"
#include "../ui/UITheme.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>

using namespace pv;

ShotLibrary::ShotLibrary(Database& database)
    : database_(database)
    , playbackSystem_(database)
    , selectedShotId_(-1)
    , currentPage_(0)
    , shotsPerPage_(12)
    , currentView_(ViewMode::Grid)
    , showingPreview_(false) {
    
    loadShots();
}

bool ShotLibrary::addShot(const ShotRecord& shot,
                         const std::string& title,
                         const std::string& description,
                         ShotCategory category,
                         ShotDifficulty difficulty) {
    
    LibraryShot libShot;
    libShot.sessionId = shot.sessionId;
    libShot.shotNumber = shot.shotNumber;
    libShot.playerId = shot.playerId;
    libShot.title = title;
    libShot.description = description;
    libShot.category = category;
    libShot.difficulty = difficulty;
    libShot.successRate = 0.0f;
    libShot.practiceCount = 0;
    libShot.cueBallStart = cv::Point2f(shot.ballX, shot.ballY);
    libShot.targetBall = cv::Point2f(shot.targetX, shot.targetY);
    libShot.shotSpeed = shot.shotSpeed;
    libShot.shotType = shot.shotType;
    libShot.isFavorite = false;
    libShot.dateRecorded = std::chrono::system_clock::now();
    
    // For now, we'll store shot library data in a simple way
    // In a full implementation, this would need its own database table
    currentResults_.push_back(libShot);
    
    return true;
}

bool ShotLibrary::removeShot(int shotId) {
    auto it = std::find_if(currentResults_.begin(), currentResults_.end(),
        [shotId](const LibraryShot& shot) { return shot.id == shotId; });
    
    if (it != currentResults_.end()) {
        currentResults_.erase(it);
        return true;
    }
    
    return false;
}

bool ShotLibrary::updateShot(const LibraryShot& shot) {
    auto it = std::find_if(currentResults_.begin(), currentResults_.end(),
        [&shot](const LibraryShot& s) { return s.id == shot.id; });
    
    if (it != currentResults_.end()) {
        *it = shot;
        return true;
    }
    
    return false;
}

std::vector<ShotLibrary::LibraryShot> ShotLibrary::searchShots(const SearchCriteria& criteria) const {
    std::vector<LibraryShot> results;
    
    for (const auto& shot : currentResults_) {
        // Category filter
        if (criteria.category != ShotCategory::All && shot.category != criteria.category) {
            continue;
        }
        
        // Difficulty filter
        if (static_cast<int>(shot.difficulty) < static_cast<int>(criteria.minDifficulty) ||
            static_cast<int>(shot.difficulty) > static_cast<int>(criteria.maxDifficulty)) {
            continue;
        }
        
        // Player filter
        if (criteria.playerId != -1 && shot.playerId != criteria.playerId) {
            continue;
        }
        
        // Favorites filter
        if (criteria.favoritesOnly && !shot.isFavorite) {
            continue;
        }
        
        // Success rate filter
        if (shot.successRate < criteria.minSuccessRate) {
            continue;
        }
        
        // Text search
        if (!criteria.searchText.empty()) {
            std::string searchLower = criteria.searchText;
            std::transform(searchLower.begin(), searchLower.end(), searchLower.begin(), ::tolower);
            
            std::string titleLower = shot.title;
            std::string descLower = shot.description;
            std::transform(titleLower.begin(), titleLower.end(), titleLower.begin(), ::tolower);
            std::transform(descLower.begin(), descLower.end(), descLower.begin(), ::tolower);
            
            if (titleLower.find(searchLower) == std::string::npos &&
                descLower.find(searchLower) == std::string::npos) {
                continue;
            }
        }
        
        results.push_back(shot);
    }
    
    // Sort results
    sortShots(results, criteria.sortBy, criteria.ascending);
    
    return results;
}

ShotLibrary::LibraryShot ShotLibrary::getShot(int shotId) const {
    auto it = std::find_if(currentResults_.begin(), currentResults_.end(),
        [shotId](const LibraryShot& shot) { return shot.id == shotId; });
    
    return (it != currentResults_.end()) ? *it : LibraryShot{};
}

std::vector<ShotLibrary::LibraryShot> ShotLibrary::getShotsByCategory(ShotCategory category) const {
    SearchCriteria criteria;
    criteria.category = category;
    return searchShots(criteria);
}

std::vector<ShotLibrary::LibraryShot> ShotLibrary::getFavoriteShots(int playerId) const {
    SearchCriteria criteria;
    criteria.favoritesOnly = true;
    criteria.playerId = playerId;
    return searchShots(criteria);
}

bool ShotLibrary::setFavorite(int shotId, bool isFavorite) {
    auto it = std::find_if(currentResults_.begin(), currentResults_.end(),
        [shotId](const LibraryShot& shot) { return shot.id == shotId; });
    
    if (it != currentResults_.end()) {
        it->isFavorite = isFavorite;
        return true;
    }
    
    return false;
}

bool ShotLibrary::addTag(int shotId, const std::string& tag) {
    auto it = std::find_if(currentResults_.begin(), currentResults_.end(),
        [shotId](const LibraryShot& shot) { return shot.id == shotId; });
    
    if (it != currentResults_.end()) {
        if (std::find(it->tags.begin(), it->tags.end(), tag) == it->tags.end()) {
            it->tags.push_back(tag);
        }
        return true;
    }
    
    return false;
}

bool ShotLibrary::removeTag(int shotId, const std::string& tag) {
    auto it = std::find_if(currentResults_.begin(), currentResults_.end(),
        [shotId](const LibraryShot& shot) { return shot.id == shotId; });
    
    if (it != currentResults_.end()) {
        auto tagIt = std::find(it->tags.begin(), it->tags.end(), tag);
        if (tagIt != it->tags.end()) {
            it->tags.erase(tagIt);
        }
        return true;
    }
    
    return false;
}

std::vector<std::string> ShotLibrary::getAllTags() const {
    std::vector<std::string> allTags;
    
    for (const auto& shot : currentResults_) {
        for (const auto& tag : shot.tags) {
            if (std::find(allTags.begin(), allTags.end(), tag) == allTags.end()) {
                allTags.push_back(tag);
            }
        }
    }
    
    std::sort(allTags.begin(), allTags.end());
    return allTags;
}

void ShotLibrary::recordPracticeAttempt(int shotId, bool successful) {
    auto it = std::find_if(currentResults_.begin(), currentResults_.end(),
        [shotId](const LibraryShot& shot) { return shot.id == shotId; });
    
    if (it != currentResults_.end()) {
        it->practiceCount++;
        it->lastPracticed = std::chrono::system_clock::now();
        
        // Update success rate (simple running average)
        if (it->practiceCount == 1) {
            it->successRate = successful ? 1.0f : 0.0f;
        } else {
            float currentTotal = it->successRate * (it->practiceCount - 1);
            it->successRate = (currentTotal + (successful ? 1.0f : 0.0f)) / it->practiceCount;
        }
    }
}

void ShotLibrary::render(cv::Mat& frame) {
    // Background
    frame = cv::Mat(720, 1280, CV_8UC3, UITheme::Colors::DarkBg);
    
    // Title bar
    UITheme::drawTitleBar(frame, "Shot Library");
    
    // View mode tabs
    std::vector<std::string> tabs = {"Grid", "List", "Search"};
    cv::Rect tabRect(0, 100, frame.cols, 60);
    UITheme::drawTabBar(frame, tabs, static_cast<int>(currentView_), tabRect);
    
    switch (currentView_) {
        case ViewMode::Grid:
            renderShotGrid(frame);
            break;
            
        case ViewMode::List:
            renderShotList(frame);
            break;
            
        case ViewMode::Details:
            if (selectedShotId_ >= 0) {
                auto shot = getShot(selectedShotId_);
                renderShotDetails(frame, shot);
            }
            break;
            
        case ViewMode::Search:
            renderSearchInterface(frame);
            break;
    }
    
    // Controls at bottom
    controlsRect_ = cv::Rect(0, frame.rows - 80, frame.cols, 80);
    renderControls(frame);
}

void ShotLibrary::renderShotDetails(cv::Mat& frame, const LibraryShot& shot) {
    detailsRect_ = cv::Rect(40, 180, frame.cols - 80, frame.rows - 300);
    UITheme::drawCard(frame, detailsRect_, UITheme::Colors::MediumBg, 180);
    
    // Shot title
    cv::putText(frame, shot.title, cv::Point(detailsRect_.x + 20, detailsRect_.y + 40),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::HeadingSize,
               UITheme::Colors::NeonCyan, UITheme::Fonts::HeadingThickness);
    
    // Category and difficulty
    int y = detailsRect_.y + 80;
    std::string catDiff = categoryToString(shot.category) + " - " + difficultyToString(shot.difficulty);
    cv::putText(frame, catDiff, cv::Point(detailsRect_.x + 20, y),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::NeonOrange, UITheme::Fonts::BodyThickness);
    
    // Description
    y += 40;
    cv::putText(frame, shot.description, cv::Point(detailsRect_.x + 20, y),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::BodyThickness);
    
    // Statistics
    y += 60;
    std::stringstream stats;
    stats << "Practice Count: " << shot.practiceCount 
          << " | Success Rate: " << std::fixed << std::setprecision(1) 
          << (shot.successRate * 100.0f) << "%";
    cv::putText(frame, stats.str(), cv::Point(detailsRect_.x + 20, y),
               UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
               UITheme::Colors::TextSecondary, 1);
    
    // Tags
    if (!shot.tags.empty()) {
        y += 40;
        cv::putText(frame, "Tags:", cv::Point(detailsRect_.x + 20, y),
                   UITheme::Fonts::FontFaceBold, UITheme::Fonts::SmallSize,
                   UITheme::Colors::TextPrimary, 1);
        
        y += 25;
        std::string tagsStr;
        for (size_t i = 0; i < shot.tags.size(); ++i) {
            tagsStr += shot.tags[i];
            if (i < shot.tags.size() - 1) tagsStr += ", ";
        }
        cv::putText(frame, tagsStr, cv::Point(detailsRect_.x + 20, y),
                   UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
                   UITheme::Colors::NeonGreen, 1);
    }
    
    // Thumbnail visualization area (right side)
    cv::Rect thumbRect(detailsRect_.x + detailsRect_.width - 300, detailsRect_.y + 20, 280, 200);
    renderShotThumbnail(frame, thumbRect, shot);
}

void ShotLibrary::onMouse(int event, int x, int y, int flags) {
    mousePos_ = cv::Point(x, y);
    
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Check tab clicks
        if (y >= 100 && y <= 160) {
            int tabWidth = 1280 / 3;
            int clickedTab = x / tabWidth;
            if (clickedTab >= 0 && clickedTab < 3) {
                currentView_ = static_cast<ViewMode>(clickedTab);
            }
        }
        
        // Check clickable areas
        for (size_t i = 0; i < clickableAreas_.size(); ++i) {
            if (clickableAreas_[i].contains(mousePos_)) {
                handleButtonClick(i);
                break;
            }
        }
    }
}

bool ShotLibrary::onKeyboard(int key) {
    switch (key) {
        case 27: // ESC - back to grid view
            currentView_ = ViewMode::Grid;
            selectedShotId_ = -1;
            return true;
            
        case 'f': // F - toggle favorite on selected shot
            if (selectedShotId_ >= 0) {
                auto shot = getShot(selectedShotId_);
                setFavorite(selectedShotId_, !shot.isFavorite);
                return true;
            }
            break;
            
        case 'p': // P - start practice mode
            if (selectedShotId_ >= 0) {
                startPractice(selectedShotId_);
                return true;
            }
            break;
    }
    
    return false;
}

bool ShotLibrary::startPractice(int shotId) {
    auto shot = getShot(shotId);
    if (shot.id == 0) return false;
    
    // This would integrate with TrainingMode to start practicing the specific shot
    // For now, just record that practice was attempted
    recordPracticeAttempt(shotId, false); // Would be updated based on actual practice result
    
    return true;
}

ShotLibrary::LibraryStats ShotLibrary::getStats() const {
    LibraryStats stats{};
    stats.totalShots = currentResults_.size();
    
    if (currentResults_.empty()) return stats;
    
    float totalDifficulty = 0.0f;
    float totalSuccessRate = 0.0f;
    std::map<std::string, int> tagCounts;
    
    // Initialize category counts
    for (int i = 0; i < 10; ++i) {
        stats.categoryCounts[i] = 0;
    }
    
    for (const auto& shot : currentResults_) {
        if (shot.isFavorite) stats.favoriteShots++;
        
        totalDifficulty += static_cast<float>(shot.difficulty);
        totalSuccessRate += shot.successRate;
        
        // Count categories
        int catIndex = static_cast<int>(shot.category);
        if (catIndex >= 0 && catIndex < 10) {
            stats.categoryCounts[catIndex]++;
        }
        
        // Count tags
        for (const auto& tag : shot.tags) {
            tagCounts[tag]++;
        }
    }
    
    stats.averageDifficulty = totalDifficulty / currentResults_.size();
    stats.averageSuccessRate = totalSuccessRate / currentResults_.size();
    
    // Find most popular tag
    if (!tagCounts.empty()) {
        auto maxTag = std::max_element(tagCounts.begin(), tagCounts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        stats.mostPopularTag = maxTag->first;
    }
    
    return stats;
}

void ShotLibrary::loadShots() {
    // Load shots from database and convert to library format
    // For now, this is simplified - would need proper shot library tables
    currentResults_.clear();
    
    // Create some example shots for demonstration
    LibraryShot example1;
    example1.id = 1;
    example1.title = "Perfect Break Shot";
    example1.description = "Powerful break with controlled spread";
    example1.category = ShotCategory::Break;
    example1.difficulty = ShotDifficulty::Intermediate;
    example1.successRate = 0.85f;
    example1.practiceCount = 20;
    example1.isFavorite = true;
    currentResults_.push_back(example1);
    
    LibraryShot example2;
    example2.id = 2;
    example2.title = "Corner Cut Shot";
    example2.description = "Precise cut shot to corner pocket";
    example2.category = ShotCategory::Cut;
    example2.difficulty = ShotDifficulty::Beginner;
    example2.successRate = 0.92f;
    example2.practiceCount = 35;
    example2.isFavorite = false;
    currentResults_.push_back(example2);
}

void ShotLibrary::applySearch() {
    currentResults_ = searchShots(currentSearch_);
}

void ShotLibrary::sortShots(std::vector<LibraryShot>& shots, int sortBy, bool ascending) const {
    switch (sortBy) {
        case 0: // Date recorded
            std::sort(shots.begin(), shots.end(), [ascending](const LibraryShot& a, const LibraryShot& b) {
                return ascending ? a.dateRecorded < b.dateRecorded : a.dateRecorded > b.dateRecorded;
            });
            break;
            
        case 1: // Difficulty
            std::sort(shots.begin(), shots.end(), [ascending](const LibraryShot& a, const LibraryShot& b) {
                return ascending ? a.difficulty < b.difficulty : a.difficulty > b.difficulty;
            });
            break;
            
        case 2: // Success rate
            std::sort(shots.begin(), shots.end(), [ascending](const LibraryShot& a, const LibraryShot& b) {
                return ascending ? a.successRate < b.successRate : a.successRate > b.successRate;
            });
            break;
    }
}

void ShotLibrary::renderSearchInterface(cv::Mat& frame) {
    searchRect_ = cv::Rect(40, 180, frame.cols - 80, 300);
    UITheme::drawCard(frame, searchRect_, UITheme::Colors::MediumBg, 180);
    
    cv::putText(frame, "Search & Filter Shots", cv::Point(searchRect_.x + 20, searchRect_.y + 40),
               UITheme::Fonts::FontFaceBold, UITheme::Fonts::BodySize,
               UITheme::Colors::TextPrimary, UITheme::Fonts::ButtonThickness);
    
    // Search interface would be implemented here
    // For now, show basic information
    cv::putText(frame, "Search functionality coming soon...", cv::Point(searchRect_.x + 20, searchRect_.y + 80),
               UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
               UITheme::Colors::TextSecondary, UITheme::Fonts::BodyThickness);
}

void ShotLibrary::renderShotGrid(cv::Mat& frame) {
    gridRect_ = cv::Rect(40, 180, frame.cols - 80, frame.rows - 300);
    clickableAreas_.clear();
    
    int cols = 4;
    int rows = 3;
    int shotWidth = (gridRect_.width - 40) / cols;
    int shotHeight = (gridRect_.height - 40) / rows;
    
    int startIdx = currentPage_ * shotsPerPage_;
    int endIdx = std::min(startIdx + shotsPerPage_, static_cast<int>(currentResults_.size()));
    
    for (int i = startIdx; i < endIdx; ++i) {
        int gridIdx = i - startIdx;
        int row = gridIdx / cols;
        int col = gridIdx % cols;
        
        cv::Rect shotRect(gridRect_.x + 20 + col * shotWidth,
                         gridRect_.y + 20 + row * shotHeight,
                         shotWidth - 20, shotHeight - 20);
        
        renderShotThumbnail(frame, shotRect, currentResults_[i]);
        clickableAreas_.push_back(shotRect);
    }
}

void ShotLibrary::renderShotList(cv::Mat& frame) {
    gridRect_ = cv::Rect(40, 180, frame.cols - 80, frame.rows - 300);
    clickableAreas_.clear();
    
    int y = gridRect_.y + 20;
    int itemHeight = 60;
    int startIdx = currentPage_ * shotsPerPage_;
    int endIdx = std::min(startIdx + shotsPerPage_, static_cast<int>(currentResults_.size()));
    
    for (int i = startIdx; i < endIdx; ++i) {
        cv::Rect itemRect(gridRect_.x + 20, y, gridRect_.width - 40, itemHeight);
        UITheme::drawCard(frame, itemRect, UITheme::Colors::DarkBg, 100);
        
        const auto& shot = currentResults_[i];
        
        // Title
        cv::putText(frame, shot.title, cv::Point(itemRect.x + 10, itemRect.y + 25),
                   UITheme::Fonts::FontFaceBold, UITheme::Fonts::BodySize,
                   UITheme::Colors::NeonCyan, UITheme::Fonts::ButtonThickness);
        
        // Category and difficulty
        std::string info = categoryToString(shot.category) + " - " + difficultyToString(shot.difficulty);
        cv::putText(frame, info, cv::Point(itemRect.x + 10, itemRect.y + 45),
                   UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
                   UITheme::Colors::TextSecondary, 1);
        
        // Success rate
        std::stringstream rate;
        rate << std::fixed << std::setprecision(0) << (shot.successRate * 100) << "%";
        cv::putText(frame, rate.str(), cv::Point(itemRect.x + itemRect.width - 80, itemRect.y + 30),
                   UITheme::Fonts::FontFace, UITheme::Fonts::BodySize,
                   UITheme::Colors::NeonGreen, UITheme::Fonts::BodyThickness);
        
        clickableAreas_.push_back(itemRect);
        y += itemHeight + 10;
    }
}

void ShotLibrary::renderControls(cv::Mat& frame) {
    UITheme::drawCard(frame, controlsRect_, UITheme::Colors::DarkBg, 200);
    
    int x = 40;
    int y = controlsRect_.y + 20;
    int buttonWidth = 100;
    int buttonHeight = 40;
    int spacing = 20;
    
    // Add shot button
    cv::Rect addButton(x, y, buttonWidth, buttonHeight);
    UITheme::drawButton(frame, "Add Shot", addButton, false, false, false);
    clickableAreas_.push_back(addButton);
    x += buttonWidth + spacing;
    
    // Page navigation
    if (currentResults_.size() > shotsPerPage_) {
        int totalPages = (currentResults_.size() + shotsPerPage_ - 1) / shotsPerPage_;
        
        if (currentPage_ > 0) {
            cv::Rect prevButton(x, y, 60, buttonHeight);
            UITheme::drawButton(frame, "Prev", prevButton, false, false, false);
            clickableAreas_.push_back(prevButton);
            x += 60 + spacing;
        }
        
        std::stringstream pageInfo;
        pageInfo << "Page " << (currentPage_ + 1) << " of " << totalPages;
        cv::putText(frame, pageInfo.str(), cv::Point(x, y + 25),
                   UITheme::Fonts::FontFace, UITheme::Fonts::SmallSize,
                   UITheme::Colors::TextSecondary, 1);
        x += 100 + spacing;
        
        if (currentPage_ < totalPages - 1) {
            cv::Rect nextButton(x, y, 60, buttonHeight);
            UITheme::drawButton(frame, "Next", nextButton, false, false, false);
            clickableAreas_.push_back(nextButton);
            x += 60 + spacing;
        }
    }
}

void ShotLibrary::renderShotThumbnail(cv::Mat& frame, const cv::Rect& rect, const LibraryShot& shot) {
    UITheme::drawCard(frame, rect, UITheme::Colors::DarkBg, 150);
    
    // Shot title
    cv::putText(frame, shot.title, cv::Point(rect.x + 5, rect.y + 20),
               UITheme::Fonts::FontFace, 0.4f, UITheme::Colors::TextPrimary, 1);
    
    // Category
    std::string category = categoryToString(shot.category);
    cv::putText(frame, category, cv::Point(rect.x + 5, rect.y + 40),
               UITheme::Fonts::FontFace, 0.3f, UITheme::Colors::NeonCyan, 1);
    
    // Difficulty stars
    int stars = static_cast<int>(shot.difficulty);
    for (int i = 0; i < 5; ++i) {
        cv::Scalar color = (i < stars) ? UITheme::Colors::NeonYellow : UITheme::Colors::TextDisabled;
        cv::circle(frame, cv::Point(rect.x + 5 + i * 15, rect.y + rect.height - 15), 3, color, -1);
    }
    
    // Favorite indicator
    if (shot.isFavorite) {
        cv::circle(frame, cv::Point(rect.x + rect.width - 15, rect.y + 15), 5, UITheme::Colors::NeonRed, -1);
    }
    
    // Success rate
    std::stringstream rate;
    rate << std::fixed << std::setprecision(0) << (shot.successRate * 100) << "%";
    cv::putText(frame, rate.str(), cv::Point(rect.x + rect.width - 40, rect.y + rect.height - 10),
               UITheme::Fonts::FontFace, 0.3f, UITheme::Colors::NeonGreen, 1);
}

void ShotLibrary::handleButtonClick(int buttonIndex) {
    // Handle button clicks based on current view and button index
    // Implementation would depend on the specific UI layout
}

std::string ShotLibrary::categoryToString(ShotCategory category) {
    switch (category) {
        case ShotCategory::Break: return "Break";
        case ShotCategory::Cut: return "Cut";
        case ShotCategory::Bank: return "Bank";
        case ShotCategory::Combination: return "Combo";
        case ShotCategory::Safety: return "Safety";
        case ShotCategory::Position: return "Position";
        case ShotCategory::Power: return "Power";
        case ShotCategory::Finesse: return "Finesse";
        case ShotCategory::Trick: return "Trick";
        case ShotCategory::Favorite: return "Favorite";
        case ShotCategory::All: return "All";
        default: return "Unknown";
    }
}

std::string ShotLibrary::difficultyToString(ShotDifficulty difficulty) {
    switch (difficulty) {
        case ShotDifficulty::Beginner: return "Beginner";
        case ShotDifficulty::Intermediate: return "Intermediate";
        case ShotDifficulty::Advanced: return "Advanced";
        case ShotDifficulty::Expert: return "Expert";
        case ShotDifficulty::Master: return "Master";
        default: return "Unknown";
    }
}