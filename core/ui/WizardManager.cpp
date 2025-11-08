#include "WizardManager.hpp"
#include <iostream>

namespace pv {

WizardManager::WizardManager()
    : currentPage_(0), running_(false), completed_(false) {
}

void WizardManager::addPage(std::unique_ptr<WizardPage> page) {
    pages_.push_back(std::move(page));
}

bool WizardManager::run() {
    if (pages_.empty()) {
        std::cerr << "No pages added to wizard!" << std::endl;
        return false;
    }
    
    // Create window
    cv::namedWindow(windowName_, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName_, 1280, 720);
    cv::setMouseCallback(windowName_, WizardManager::onMouse, this);
    
    running_ = true;
    currentPage_ = 0;
    pages_[currentPage_]->init();
    
    while (running_) {
        // Get camera frame if camera is open
        cv::Mat frame;
        if (camera_.isOpened()) {
            camera_ >> frame;
        }
        
        // Render current page
        renderCurrentPage(frame);
        
        // Display
        if (!displayImage_.empty()) {
            cv::imshow(windowName_, displayImage_);
        }
        
        // Handle keyboard
        int key = cv::waitKey(30);
        if (key != -1) {
            handleKeyboard(key);
        }
    }
    
    cv::destroyWindow(windowName_);
    
    if (completed_) {
        return saveConfig();
    }
    
    return false;
}

void WizardManager::renderCurrentPage(const cv::Mat& frame) {
    if (currentPage_ >= pages_.size()) {
        return;
    }
    
    displayImage_ = pages_[currentPage_]->render(frame, config_);
    
    // Draw navigation buttons
    int buttonWidth = 120;
    int buttonHeight = 40;
    int margin = 20;
    int buttonY = displayImage_.rows - margin - buttonHeight;
    
    // Back button (disabled on first page)
    cv::Rect backButton(margin, buttonY, buttonWidth, buttonHeight);
    bool backEnabled = currentPage_ > 0;
    pages_[currentPage_]->drawButton(displayImage_, "< Back", backButton, false, backEnabled);
    
    // Next/Finish button
    int nextX = displayImage_.cols - margin - buttonWidth;
    cv::Rect nextButton(nextX, buttonY, buttonWidth, buttonHeight);
    bool isLastPage = currentPage_ == pages_.size() - 1;
    bool nextEnabled = pages_[currentPage_]->isComplete();
    std::string nextText = isLastPage ? "Finish" : "Next >";
    pages_[currentPage_]->drawButton(displayImage_, nextText, nextButton, true, nextEnabled);
    
    // Progress indicator
    pages_[currentPage_]->drawProgressBar(displayImage_, 
                                         static_cast<int>(currentPage_ + 1), 
                                         static_cast<int>(pages_.size()));
    
    // Help text
    std::string helpText = pages_[currentPage_]->getHelpText() + " | ESC to cancel";
    pages_[currentPage_]->drawHelpBar(displayImage_, helpText);
}

void WizardManager::handleMouse(int event, int x, int y, int flags) {
    if (currentPage_ >= pages_.size()) {
        return;
    }
    
    // Check navigation buttons
    int buttonWidth = 120;
    int buttonHeight = 40;
    int margin = 20;
    int buttonY = displayImage_.rows - margin - buttonHeight;
    
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Back button
        cv::Rect backButton(margin, buttonY, buttonWidth, buttonHeight);
        if (backButton.contains(cv::Point(x, y)) && currentPage_ > 0) {
            previousPage();
            return;
        }
        
        // Next button
        int nextX = displayImage_.cols - margin - buttonWidth;
        cv::Rect nextButton(nextX, buttonY, buttonWidth, buttonHeight);
        if (nextButton.contains(cv::Point(x, y)) && pages_[currentPage_]->isComplete()) {
            if (currentPage_ == pages_.size() - 1) {
                // Finish
                completed_ = true;
                running_ = false;
            } else {
                nextPage();
            }
            return;
        }
    }
    
    // Forward to current page
    pages_[currentPage_]->onMouse(event, x, y, flags);
}

void WizardManager::handleKeyboard(int key) {
    // ESC to cancel
    if (key == 27) {
        running_ = false;
        completed_ = false;
        return;
    }
    
    // Forward to current page
    if (currentPage_ < pages_.size()) {
        pages_[currentPage_]->onKey(key);
    }
}

void WizardManager::nextPage() {
    if (currentPage_ >= pages_.size() - 1) {
        return;
    }
    
    // Validate current page
    std::string error = pages_[currentPage_]->validate();
    if (!error.empty()) {
        std::cerr << "Validation error: " << error << std::endl;
        return;
    }
    
    currentPage_++;
    pages_[currentPage_]->init();
    
    // Open camera if not open and we need it
    if (!camera_.isOpened()) {
        camera_.open(config_.cameraIndex);
    }
}

void WizardManager::previousPage() {
    if (currentPage_ == 0) {
        return;
    }
    
    currentPage_--;
    pages_[currentPage_]->init();
}

void WizardManager::onMouse(int event, int x, int y, int flags, void* userdata) {
    WizardManager* manager = static_cast<WizardManager*>(userdata);
    manager->handleMouse(event, x, y, flags);
}

bool WizardManager::saveConfig() {
    // TODO: Implement actual file saving
    // For now, just print configuration
    std::cout << "Saving configuration..." << std::endl;
    std::cout << "  Camera: " << config_.cameraIndex << std::endl;
    std::cout << "  Rotation: " << config_.rotation << std::endl;
    std::cout << "  Table size: " << config_.tableWidth << "m x " << config_.tableLength << "m" << std::endl;
    std::cout << "  Corners: " << config_.tableCorners.size() << std::endl;
    std::cout << "  Pockets: " << config_.pocketPositions.size() << std::endl;
    
    return true;
}

} // namespace pv
