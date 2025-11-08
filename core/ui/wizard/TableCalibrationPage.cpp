#include "TableCalibrationPage.hpp"
#include <iostream>

namespace pv {

TableCalibrationPage::TableCalibrationPage()
    : selectedCorner_(-1), hoveredCorner_(-1), isDragging_(false),
      homographyValid_(false), previewRect_(750, 160, 500, 400) {
}

void TableCalibrationPage::init() {
    corners_.clear();
    // Initialize with default positions (centered rectangle)
    corners_.push_back(cv::Point2f(150, 150));   // TL
    corners_.push_back(cv::Point2f(550, 150));   // TR
    corners_.push_back(cv::Point2f(550, 450));   // BR
    corners_.push_back(cv::Point2f(150, 450));   // BL
    
    selectedCorner_ = -1;
    hoveredCorner_ = -1;
    isDragging_ = false;
    homographyValid_ = false;
}

cv::Mat TableCalibrationPage::render(const cv::Mat& frame, WizardConfig& config) {
    cv::Mat display(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));
    
    drawTitle(display, getTitle());
    
    // Instructions
    std::string instructions = "Click and drag the corners to mark the table boundaries:";
    cv::putText(display, instructions, cv::Point(40, 120),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    
    // Camera preview with overlay (left side)
    if (!frame.empty()) {
        cv::Rect frameRect(40, 160, 640, 480);
        cv::Mat resized;
        cv::resize(frame, resized, frameRect.size());
        resized.copyTo(display(frameRect));
        
        // Draw border
        cv::rectangle(display, frameRect, cv::Scalar(100, 100, 100), 2);
        
        // Scale corners to display coordinates
        std::vector<cv::Point2f> displayCorners;
        float scaleX = 640.0f / frame.cols;
        float scaleY = 480.0f / frame.rows;
        
        for (const auto& corner : corners_) {
            displayCorners.push_back(cv::Point2f(
                frameRect.x + corner.x * scaleX,
                frameRect.y + corner.y * scaleY
            ));
        }
        
        // Draw table outline
        if (displayCorners.size() == 4) {
            for (size_t i = 0; i < 4; i++) {
                size_t next = (i + 1) % 4;
                cv::line(display, displayCorners[i], displayCorners[next],
                        cv::Scalar(100, 200, 255), 2, cv::LINE_AA);
            }
            
            // Fill with semi-transparent overlay
            std::vector<cv::Point> polyCorners;
            for (const auto& pt : displayCorners) {
                polyCorners.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
            }
            cv::Mat overlay = display.clone();
            cv::fillConvexPoly(overlay, polyCorners, cv::Scalar(100, 200, 255));
            cv::addWeighted(display, 0.7, overlay, 0.3, 0, display);
        }
        
        // Draw corner handles
        for (size_t i = 0; i < displayCorners.size(); i++) {
            cv::Scalar color = cv::Scalar(100, 200, 255);
            int radius = 10;
            
            if (static_cast<int>(i) == selectedCorner_) {
                color = cv::Scalar(0, 255, 0);
                radius = 12;
            } else if (static_cast<int>(i) == hoveredCorner_) {
                color = cv::Scalar(150, 220, 255);
                radius = 11;
            }
            
            cv::circle(display, displayCorners[i], radius, color, -1, cv::LINE_AA);
            cv::circle(display, displayCorners[i], radius + 2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            
            // Draw label
            std::string label = std::to_string(i + 1);
            cv::putText(display, label, displayCorners[i] - cv::Point2f(5, -5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
        
        // Compute homography if we have 4 corners
        if (corners_.size() == 4) {
            computeHomography();
            config.tableCorners = corners_;
            config.homographyMatrix = homography_;
        }
    }
    
    // Right side - Preview and instructions
    int infoX = 720;
    int infoY = 180;
    
    cv::putText(display, "Corner Order:", cv::Point(infoX, infoY),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    infoY += 30;
    
    for (size_t i = 0; i < cornerNames_.size(); i++) {
        std::string text = std::to_string(i + 1) + ". " + cornerNames_[i];
        cv::Scalar color = static_cast<int>(i) == selectedCorner_ ? 
                          cv::Scalar(100, 255, 100) : cv::Scalar(200, 200, 200);
        cv::putText(display, text, cv::Point(infoX + 10, infoY),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
        infoY += 25;
    }
    
    infoY += 20;
    
    // Status
    if (homographyValid_) {
        cv::putText(display, "Status: Ready", cv::Point(infoX, infoY),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(100, 255, 100), 1, cv::LINE_AA);
        infoY += 40;
        
        // Draw transformed preview
        if (!frame.empty() && !homography_.empty()) {
            drawTransformedPreview(display, frame);
        }
    } else {
        cv::putText(display, "Status: Adjusting", cv::Point(infoX, infoY),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 100), 1, cv::LINE_AA);
        infoY += 40;
    }
    
    return display;
}

void TableCalibrationPage::onMouse(int event, int x, int y, int flags) {
    cv::Point mousePos(x, y);
    
    // Only handle clicks in the camera preview area
    cv::Rect frameRect(40, 160, 640, 480);
    if (!frameRect.contains(mousePos)) {
        hoveredCorner_ = -1;
        return;
    }
    
    // Convert to frame coordinates
    float frameX = (x - frameRect.x) / (640.0f / 640.0f);  // Adjust based on actual frame size
    float frameY = (y - frameRect.y) / (480.0f / 480.0f);
    cv::Point framePos(static_cast<int>(frameX), static_cast<int>(frameY));
    
    if (event == cv::EVENT_MOUSEMOVE) {
        if (isDragging_ && selectedCorner_ >= 0) {
            // Update corner position
            corners_[selectedCorner_] = cv::Point2f(framePos);
            homographyValid_ = false;
        } else {
            // Check for hover
            hoveredCorner_ = findNearestCorner(framePos, 30.0f);
        }
    }
    else if (event == cv::EVENT_LBUTTONDOWN) {
        selectedCorner_ = findNearestCorner(framePos, 30.0f);
        if (selectedCorner_ >= 0) {
            isDragging_ = true;
            dragStart_ = framePos;
        }
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        isDragging_ = false;
        if (corners_.size() == 4) {
            computeHomography();
        }
    }
}

bool TableCalibrationPage::onKey(int key) {
    // Arrow keys for fine adjustment
    if (selectedCorner_ >= 0) {
        float delta = 1.0f;
        if (key == 82) { // Up arrow
            corners_[selectedCorner_].y -= delta;
            homographyValid_ = false;
            return true;
        } else if (key == 84) { // Down arrow
            corners_[selectedCorner_].y += delta;
            homographyValid_ = false;
            return true;
        } else if (key == 81) { // Left arrow
            corners_[selectedCorner_].x -= delta;
            homographyValid_ = false;
            return true;
        } else if (key == 83) { // Right arrow
            corners_[selectedCorner_].x += delta;
            homographyValid_ = false;
            return true;
        }
    }
    
    // Number keys to select corner
    if (key >= '1' && key <= '4') {
        selectedCorner_ = key - '1';
        return true;
    }
    
    return false;
}

bool TableCalibrationPage::isComplete() const {
    return corners_.size() == 4 && homographyValid_;
}

std::string TableCalibrationPage::getTitle() const {
    return "Table Calibration";
}

std::string TableCalibrationPage::getHelpText() const {
    return "Drag corners to match table edges. Use arrow keys for fine adjustment. Press 1-4 to select corners.";
}

std::string TableCalibrationPage::validate() const {
    if (corners_.size() != 4) {
        return "Please mark all 4 table corners";
    }
    
    if (!homographyValid_) {
        return "Invalid table calibration. Please adjust corners.";
    }
    
    return "";
}

int TableCalibrationPage::findNearestCorner(cv::Point pos, float threshold) {
    int nearest = -1;
    float minDist = threshold;
    
    for (size_t i = 0; i < corners_.size(); i++) {
        float dist = cv::norm(corners_[i] - cv::Point2f(pos));
        if (dist < minDist) {
            minDist = dist;
            nearest = static_cast<int>(i);
        }
    }
    
    return nearest;
}

void TableCalibrationPage::computeHomography() {
    if (corners_.size() != 4) {
        homographyValid_ = false;
        return;
    }
    
    // Define destination points (normalized table coordinates)
    // We'll use a standard aspect ratio
    float width = 400.0f;
    float height = 600.0f;
    
    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(0, 0),              // TL
        cv::Point2f(width, 0),          // TR
        cv::Point2f(width, height),     // BR
        cv::Point2f(0, height)          // BL
    };
    
    try {
        homography_ = cv::findHomography(corners_, dstPoints);
        homographyValid_ = !homography_.empty();
    } catch (const cv::Exception& e) {
        std::cerr << "Homography computation failed: " << e.what() << std::endl;
        homographyValid_ = false;
    }
}

void TableCalibrationPage::drawTransformedPreview(cv::Mat& display, const cv::Mat& frame) {
    if (homography_.empty()) {
        return;
    }
    
    try {
        cv::Mat warped;
        cv::warpPerspective(frame, warped, homography_, cv::Size(400, 600));
        
        // Draw preview
        cv::Rect previewRect(750, 320, 300, 300);
        cv::Mat resized;
        cv::resize(warped, resized, previewRect.size());
        
        cv::rectangle(display, previewRect, cv::Scalar(100, 100, 100), 2);
        resized.copyTo(display(previewRect));
        
        // Label
        cv::putText(display, "Bird's Eye View", cv::Point(750, 310),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    } catch (const cv::Exception& e) {
        std::cerr << "Warp failed: " << e.what() << std::endl;
    }
}

} // namespace pv
