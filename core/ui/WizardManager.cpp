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
        // TODO: Show error dialog in UI
        return;
    }
    
    currentPage_++;
    pages_[currentPage_]->init();
    
    // Open camera if not open and we need it
    if (!camera_.isOpened()) {
        camera_.open(config_.cameraIndex);
        if (camera_.isOpened() && config_.resolution.width > 0) {
            camera_.set(cv::CAP_PROP_FRAME_WIDTH, config_.resolution.width);
            camera_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.resolution.height);
        }
    }
}

bool WizardManager::validateConfig() const {
    // Validate camera index
    if (config_.cameraIndex < 0) {
        std::cerr << "Invalid camera index: " << config_.cameraIndex << std::endl;
        return false;
    }
    
    // Validate rotation
    if (config_.rotation != 0 && config_.rotation != 90 && 
        config_.rotation != 180 && config_.rotation != 270) {
        std::cerr << "Invalid rotation: " << config_.rotation << std::endl;
        return false;
    }
    
    // Validate table dimensions
    if (config_.tableWidth <= 0 || config_.tableLength <= 0) {
        std::cerr << "Invalid table dimensions: " << config_.tableWidth 
                  << " x " << config_.tableLength << std::endl;
        return false;
    }
    
    // Validate table corners
    if (config_.tableCorners.size() != 4) {
        std::cerr << "Invalid number of table corners: " << config_.tableCorners.size() << std::endl;
        return false;
    }
    
    // Validate homography matrix
    if (config_.homographyMatrix.empty()) {
        std::cerr << "Homography matrix is empty" << std::endl;
        return false;
    }
    
    if (config_.homographyMatrix.rows != 3 || config_.homographyMatrix.cols != 3) {
        std::cerr << "Invalid homography matrix dimensions" << std::endl;
        return false;
    }
    
    // Check that homography is not identity (should be calibrated)
    bool isIdentity = true;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(config_.homographyMatrix.at<double>(i, j) - expected) > 0.01) {
                isIdentity = false;
                break;
            }
        }
        if (!isIdentity) break;
    }
    
    if (isIdentity) {
        std::cerr << "Warning: Homography is identity matrix (not calibrated)" << std::endl;
        // This is a warning, not an error - allow proceeding
    }
    
    return true;
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
    std::cout << "Saving configuration..." << std::endl;
    
    try {
        // Create config directory if it doesn't exist
        cv::utils::fs::createDirectories("config");
        
        // Save camera config
        cv::FileStorage cameraFs(config_.cameraConfigPath, cv::FileStorage::WRITE);
        if (!cameraFs.isOpened()) {
            std::cerr << "Failed to open camera config file for writing" << std::endl;
            return false;
        }
        
        // Apply rotation/flip transformations to resolution if needed
        int width = config_.resolution.width;
        int height = config_.resolution.height;
        if (config_.rotation == 90 || config_.rotation == 270) {
            std::swap(width, height);
        }
        
        cameraFs << "width" << width;
        cameraFs << "height" << height;
        cameraFs << "fps" << 60;
        cameraFs << "device_index" << config_.cameraIndex;
        cameraFs << "rotation" << config_.rotation;
        cameraFs << "flip_horizontal" << config_.flipHorizontal;
        cameraFs << "flip_vertical" << config_.flipVertical;
        cameraFs.release();
        std::cout << "  ✓ Saved: " << config_.cameraConfigPath << std::endl;
        
        // Save table config
        cv::FileStorage tableFs(config_.tableConfigPath, cv::FileStorage::WRITE);
        if (!tableFs.isOpened()) {
            std::cerr << "Failed to open table config file for writing" << std::endl;
            return false;
        }
        
        // Convert dimensions to mm (config uses meters, file uses mm)
        tableFs << "table_width" << static_cast<int>(config_.tableLength * 1000);
        tableFs << "table_height" << static_cast<int>(config_.tableWidth * 1000);
        tableFs << "ball_radius_px" << 10;
        
        // Save homography as flat array
        if (!config_.homographyMatrix.empty() && 
            config_.homographyMatrix.rows == 3 && 
            config_.homographyMatrix.cols == 3) {
            std::vector<double> homography_flat;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    homography_flat.push_back(config_.homographyMatrix.at<double>(i, j));
                }
            }
            tableFs << "homography" << homography_flat;
        } else {
            // Identity matrix as fallback
            tableFs << "homography" << std::vector<double>{1,0,0,0,1,0,0,0,1};
        }
        
        // Save pockets if available
        if (!config_.pocketPositions.empty()) {
            tableFs << "pockets" << "[";
            for (const auto& pos : config_.pocketPositions) {
                tableFs << "[";
                // Create small square around pocket center
                int radius = config_.pocketRadii.empty() ? 50 : 
                            static_cast<int>(config_.pocketRadii[0]);
                tableFs << "[" << static_cast<int>(pos.x - radius) 
                       << static_cast<int>(pos.y - radius) << "]";
                tableFs << "[" << static_cast<int>(pos.x + radius) 
                       << static_cast<int>(pos.y - radius) << "]";
                tableFs << "[" << static_cast<int>(pos.x + radius) 
                       << static_cast<int>(pos.y + radius) << "]";
                tableFs << "[" << static_cast<int>(pos.x - radius) 
                       << static_cast<int>(pos.y + radius) << "]";
                tableFs << "]";
            }
            tableFs << "]";
        }
        
        tableFs.release();
        std::cout << "  ✓ Saved: " << config_.tableConfigPath << std::endl;
        
        // Save colors config (use existing or create default)
        cv::FileStorage colorsFs(config_.colorsConfigPath, cv::FileStorage::WRITE);
        if (!colorsFs.isOpened()) {
            std::cerr << "Failed to open colors config file for writing" << std::endl;
            return false;
        }
        
        colorsFs << "prototypes" << "{";
        if (config_.ballColors.empty()) {
            // Write default colors
            colorsFs << "cue" << std::vector<float>{70, -5, 20};
            colorsFs << "1" << std::vector<float>{50, 10, 20};
            colorsFs << "2" << std::vector<float>{40, -10, 30};
            colorsFs << "3" << std::vector<float>{40, 20, 20};
            colorsFs << "4" << std::vector<float>{45, 5, -20};
            colorsFs << "5" << std::vector<float>{45, 25, 10};
            colorsFs << "6" << std::vector<float>{35, -5, -10};
            colorsFs << "7" << std::vector<float>{30, 15, 5};
            colorsFs << "8" << std::vector<float>{20, 0, 0};
            colorsFs << "9" << std::vector<float>{55, 10, 10};
            colorsFs << "10" << std::vector<float>{60, -5, 5};
            colorsFs << "11" << std::vector<float>{45, 12, 22};
            colorsFs << "12" << std::vector<float>{35, -12, 18};
            colorsFs << "13" << std::vector<float>{40, 22, -8};
            colorsFs << "14" << std::vector<float>{48, 4, -10};
            colorsFs << "15" << std::vector<float>{42, 16, 14};
        } else {
            // Write user-calibrated colors
            for (const auto& [label, color] : config_.ballColors) {
                std::string key = (label == 0) ? "cue" : std::to_string(label);
                colorsFs << key << std::vector<float>{color[0], color[1], color[2]};
            }
        }
        colorsFs << "}";
        colorsFs.release();
        std::cout << "  ✓ Saved: " << config_.colorsConfigPath << std::endl;
        
        std::cout << "\n✓ Configuration saved successfully!" << std::endl;
        std::cout << "\nConfiguration Summary:" << std::endl;
        std::cout << "  Camera: Device " << config_.cameraIndex << std::endl;
        std::cout << "  Rotation: " << config_.rotation << "°";
        if (config_.flipHorizontal || config_.flipVertical) {
            std::cout << " (";
            if (config_.flipHorizontal) std::cout << "H-flip";
            if (config_.flipHorizontal && config_.flipVertical) std::cout << ", ";
            if (config_.flipVertical) std::cout << "V-flip";
            std::cout << ")";
        }
        std::cout << std::endl;
        std::cout << "  Table: " << config_.tableLength << "m x " << config_.tableWidth << "m" << std::endl;
        std::cout << "  Corners: " << config_.tableCorners.size() << " marked" << std::endl;
        if (!config_.pocketPositions.empty()) {
            std::cout << "  Pockets: " << config_.pocketPositions.size() << " configured" << std::endl;
        }
        
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception while saving config: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception while saving config: " << e.what() << std::endl;
        return false;
    }
}

} // namespace pv
