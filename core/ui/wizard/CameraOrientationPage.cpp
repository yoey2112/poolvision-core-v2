#include "CameraOrientationPage.hpp"

namespace pv {

CameraOrientationPage::CameraOrientationPage()
    : rotation_(0), flipHorizontal_(false), flipVertical_(false),
      hoveredRotation_(-1), userModified_(false) {
}

void CameraOrientationPage::init() {
    rotation_ = 0;
    flipHorizontal_ = false;
    flipVertical_ = false;
    userModified_ = false;
}

cv::Mat CameraOrientationPage::render(const cv::Mat& frame, WizardConfig& config) {
    cv::Mat display(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));
    
    drawTitle(display, getTitle());
    
    // Instructions
    std::string instructions = "Adjust the camera orientation to match your setup:";
    cv::putText(display, instructions, cv::Point(40, 120),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    
    // Camera preview (left side)
    if (!frame.empty()) {
        cv::Mat transformed = applyTransforms(frame, rotation_, flipHorizontal_, flipVertical_);
        
        // Scale to fit preview area
        cv::Rect previewRect(40, 160, 640, 480);
        cv::Mat resized;
        cv::resize(transformed, resized, previewRect.size());
        
        // Draw preview border
        cv::rectangle(display, previewRect, cv::Scalar(100, 100, 100), 2);
        resized.copyTo(display(previewRect));
        
        // Draw preview label
        cv::putText(display, "Live Preview", cv::Point(previewRect.x, previewRect.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    }
    
    // Controls (right side)
    int controlX = 720;
    int controlY = 160;
    
    // Rotation controls
    cv::putText(display, "Rotation:", cv::Point(controlX, controlY),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    controlY += 40;
    
    int rotations[] = {0, 90, 180, 270};
    for (int i = 0; i < 4; i++) {
        cv::Rect btnRect = getRotationButtonRect(rotations[i]);
        bool isSelected = (rotation_ == rotations[i]);
        bool isHovered = (hoveredRotation_ == rotations[i]);
        
        std::string label = std::to_string(rotations[i]) + "\u00b0";
        drawButton(display, label, btnRect, isHovered || isSelected, true);
        
        // Draw checkmark if selected
        if (isSelected) {
            cv::circle(display, cv::Point(btnRect.x + 20, btnRect.y + btnRect.height / 2),
                      6, cv::Scalar(100, 255, 100), -1);
        }
    }
    
    controlY += 100;
    
    // Flip controls
    cv::putText(display, "Flip:", cv::Point(controlX, controlY),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    controlY += 40;
    
    // Horizontal flip checkbox
    cv::Rect flipHRect = getFlipButtonRect("horizontal");
    drawCheckbox(display, "Flip Horizontal", cv::Point(flipHRect.x, flipHRect.y), flipHorizontal_);
    
    // Vertical flip checkbox
    cv::Rect flipVRect = getFlipButtonRect("vertical");
    drawCheckbox(display, "Flip Vertical", cv::Point(flipVRect.x, flipVRect.y), flipVertical_);
    
    controlY += 120;
    
    // Current orientation display
    cv::putText(display, "Current Settings:", cv::Point(controlX, controlY),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    controlY += 30;
    
    std::string rotText = "  Rotation: " + std::to_string(rotation_) + "\u00b0";
    cv::putText(display, rotText, cv::Point(controlX, controlY),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    controlY += 25;
    
    std::string flipHText = "  Horizontal: " + std::string(flipHorizontal_ ? "Yes" : "No");
    cv::putText(display, flipHText, cv::Point(controlX, controlY),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    controlY += 25;
    
    std::string flipVText = "  Vertical: " + std::string(flipVertical_ ? "Yes" : "No");
    cv::putText(display, flipVText, cv::Point(controlX, controlY),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    
    // Update config
    config.rotation = rotation_;
    config.flipHorizontal = flipHorizontal_;
    config.flipVertical = flipVertical_;
    
    return display;
}

void CameraOrientationPage::onMouse(int event, int x, int y, int flags) {
    if (event == cv::EVENT_MOUSEMOVE) {
        hoveredRotation_ = -1;
        hoveredFlip_ = "";
        
        // Check rotation buttons
        int rotations[] = {0, 90, 180, 270};
        for (int i = 0; i < 4; i++) {
            cv::Rect rect = getRotationButtonRect(rotations[i]);
            if (rect.contains(cv::Point(x, y))) {
                hoveredRotation_ = rotations[i];
                break;
            }
        }
        
        // Check flip buttons
        if (getFlipButtonRect("horizontal").contains(cv::Point(x, y))) {
            hoveredFlip_ = "horizontal";
        } else if (getFlipButtonRect("vertical").contains(cv::Point(x, y))) {
            hoveredFlip_ = "vertical";
        }
    }
    else if (event == cv::EVENT_LBUTTONDOWN) {
        // Check rotation buttons
        int rotations[] = {0, 90, 180, 270};
        for (int i = 0; i < 4; i++) {
            cv::Rect rect = getRotationButtonRect(rotations[i]);
            if (rect.contains(cv::Point(x, y))) {
                rotation_ = rotations[i];
                userModified_ = true;
                return;
            }
        }
        
        // Check flip horizontal checkbox
        cv::Rect flipHRect = getFlipButtonRect("horizontal");
        cv::Rect checkboxH(flipHRect.x, flipHRect.y, 20, 20);
        if (checkboxH.contains(cv::Point(x, y))) {
            flipHorizontal_ = !flipHorizontal_;
            userModified_ = true;
            return;
        }
        
        // Check flip vertical checkbox
        cv::Rect flipVRect = getFlipButtonRect("vertical");
        cv::Rect checkboxV(flipVRect.x, flipVRect.y, 20, 20);
        if (checkboxV.contains(cv::Point(x, y))) {
            flipVertical_ = !flipVertical_;
            userModified_ = true;
            return;
        }
    }
}

bool CameraOrientationPage::onKey(int key) {
    switch (key) {
        case 'r':
        case 'R':
            // Rotate 90 degrees
            rotation_ = (rotation_ + 90) % 360;
            userModified_ = true;
            return true;
        case 'h':
        case 'H':
            // Flip horizontal
            flipHorizontal_ = !flipHorizontal_;
            userModified_ = true;
            return true;
        case 'v':
        case 'V':
            // Flip vertical
            flipVertical_ = !flipVertical_;
            userModified_ = true;
            return true;
    }
    return false;
}

bool CameraOrientationPage::isComplete() const {
    // Always complete - user can skip or modify
    return true;
}

std::string CameraOrientationPage::getTitle() const {
    return "Camera Orientation";
}

std::string CameraOrientationPage::getHelpText() const {
    return "Press R to rotate, H for horizontal flip, V for vertical flip. Click buttons to adjust.";
}

std::string CameraOrientationPage::validate() const {
    // No validation needed - any orientation is valid
    return "";
}

cv::Mat CameraOrientationPage::applyTransforms(const cv::Mat& input, int rotation, 
                                              bool flipH, bool flipV) {
    if (input.empty()) {
        return input;
    }
    
    cv::Mat result = input.clone();
    
    // Apply rotation
    if (rotation == 90) {
        cv::rotate(result, result, cv::ROTATE_90_CLOCKWISE);
    } else if (rotation == 180) {
        cv::rotate(result, result, cv::ROTATE_180);
    } else if (rotation == 270) {
        cv::rotate(result, result, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    
    // Apply flips
    if (flipH && flipV) {
        cv::flip(result, result, -1);
    } else if (flipH) {
        cv::flip(result, result, 1);
    } else if (flipV) {
        cv::flip(result, result, 0);
    }
    
    return result;
}

cv::Rect CameraOrientationPage::getRotationButtonRect(int rotation) const {
    int controlX = 720;
    int controlY = 200;
    int buttonWidth = 100;
    int buttonHeight = 40;
    int spacing = 10;
    
    int index = rotation / 90;
    int x = controlX + (index % 2) * (buttonWidth + spacing);
    int y = controlY + (index / 2) * (buttonHeight + spacing);
    
    return cv::Rect(x, y, buttonWidth, buttonHeight);
}

cv::Rect CameraOrientationPage::getFlipButtonRect(const std::string& type) const {
    int controlX = 720;
    int controlY = type == "horizontal" ? 340 : 370;
    
    return cv::Rect(controlX, controlY, 20, 20);
}

} // namespace pv
