#include "CameraSelectionPage.hpp"
#include <iostream>

namespace pv {

CameraSelectionPage::CameraSelectionPage()
    : selectedCamera_(-1), hoveredCamera_(-1), isTestingCamera_(false) {
}

void CameraSelectionPage::init() {
    enumerateCameras();
    selectedCamera_ = -1;
    isTestingCamera_ = false;
}

void CameraSelectionPage::enumerateCameras() {
    cameras_.clear();
    
    // Try to find cameras (usually 0-4 are sufficient)
    for (int i = 0; i < 5; i++) {
        cv::VideoCapture cap(i);
        if (cap.isOpened()) {
            CameraInfo info;
            info.index = i;
            info.name = "Camera " + std::to_string(i);
            info.resolution = cv::Size(
                static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
            );
            info.available = true;
            
            // Capture thumbnail
            cv::Mat frame;
            cap >> frame;
            if (!frame.empty()) {
                cv::resize(frame, info.thumbnail, 
                          cv::Size(cameraBoxWidth_ - 20, 100));
            } else {
                info.thumbnail = cv::Mat(100, cameraBoxWidth_ - 20, CV_8UC3, cv::Scalar(50, 50, 50));
            }
            
            cameras_.push_back(info);
            cap.release();
            
            // Small delay between camera checks
            cv::waitKey(100);
        }
    }
    
    if (cameras_.empty()) {
        std::cerr << "No cameras found!" << std::endl;
    }
}

cv::Mat CameraSelectionPage::render(const cv::Mat& frame, WizardConfig& config) {
    // Create display image
    cv::Mat display(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));
    
    // Draw title
    drawTitle(display, getTitle());
    
    // Draw instructions
    std::string instructions = "Select a camera to use for pool table detection:";
    cv::putText(display, instructions, cv::Point(40, 120),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    
    // Draw camera boxes
    int startY = 160;
    int startX = 40;
    
    for (size_t i = 0; i < cameras_.size(); i++) {
        cv::Rect rect = getCameraRect(i);
        
        // Determine box color
        cv::Scalar boxColor = cv::Scalar(60, 60, 60);
        if (static_cast<int>(i) == selectedCamera_) {
            boxColor = cv::Scalar(100, 200, 255);
        } else if (static_cast<int>(i) == hoveredCamera_) {
            boxColor = cv::Scalar(80, 80, 100);
        }
        
        // Draw box
        cv::rectangle(display, rect, boxColor, -1);
        cv::rectangle(display, rect, 
                     static_cast<int>(i) == selectedCamera_ ? 
                     cv::Scalar(150, 220, 255) : cv::Scalar(100, 100, 100), 2);
        
        // Draw thumbnail
        if (!cameras_[i].thumbnail.empty()) {
            cv::Rect thumbRect(rect.x + 10, rect.y + 10, 
                              cameraBoxWidth_ - 20, 100);
            cameras_[i].thumbnail.copyTo(display(thumbRect));
        }
        
        // Draw camera info
        int textY = rect.y + 120;
        cv::putText(display, cameras_[i].name, 
                   cv::Point(rect.x + 10, textY),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        
        std::string resText = std::to_string(cameras_[i].resolution.width) + "x" +
                             std::to_string(cameras_[i].resolution.height);
        cv::putText(display, resText,
                   cv::Point(rect.x + 10, textY + 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        
        // Test button
        cv::Rect testButton(rect.x + 10, rect.y + rect.height - 40, 
                           cameraBoxWidth_ - 20, 30);
        drawButton(display, "Test", testButton, 
                  static_cast<int>(i) == hoveredCamera_);
    }
    
    // If testing camera, show live preview
    if (isTestingCamera_ && testCapture_.isOpened()) {
        testCapture_ >> testFrame_;
        if (!testFrame_.empty()) {
            cv::Rect previewRect(450, 160, 800, 450);
            cv::rectangle(display, previewRect, cv::Scalar(0, 0, 0), 2);
            
            cv::Mat resized;
            cv::resize(testFrame_, resized, previewRect.size());
            resized.copyTo(display(previewRect));
            
            // Stop test button
            cv::Rect stopButton(previewRect.x + previewRect.width / 2 - 60,
                               previewRect.y + previewRect.height + 20, 120, 40);
            drawButton(display, "Stop Test", stopButton, true);
        }
    } else if (selectedCamera_ >= 0) {
        // Show selection confirmation
        std::string confirmText = "Selected: " + cameras_[selectedCamera_].name;
        cv::putText(display, confirmText, cv::Point(450, 300),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 200, 255), 2, cv::LINE_AA);
    }
    
    config.cameraIndex = selectedCamera_;
    
    return display;
}

void CameraSelectionPage::onMouse(int event, int x, int y, int flags) {
    if (event == cv::EVENT_MOUSEMOVE) {
        hoveredCamera_ = -1;
        
        // Check which camera is hovered
        for (size_t i = 0; i < cameras_.size(); i++) {
            cv::Rect rect = getCameraRect(i);
            if (rect.contains(cv::Point(x, y))) {
                hoveredCamera_ = i;
                break;
            }
        }
    }
    else if (event == cv::EVENT_LBUTTONDOWN) {
        // Check test buttons
        for (size_t i = 0; i < cameras_.size(); i++) {
            cv::Rect rect = getCameraRect(i);
            cv::Rect testButton(rect.x + 10, rect.y + rect.height - 40,
                               cameraBoxWidth_ - 20, 30);
            
            if (testButton.contains(cv::Point(x, y))) {
                testCamera(i);
                return;
            }
        }
        
        // Check camera selection
        for (size_t i = 0; i < cameras_.size(); i++) {
            cv::Rect rect = getCameraRect(i);
            if (rect.contains(cv::Point(x, y))) {
                selectedCamera_ = i;
                isTestingCamera_ = false;
                if (testCapture_.isOpened()) {
                    testCapture_.release();
                }
                return;
            }
        }
        
        // Check stop test button
        if (isTestingCamera_) {
            cv::Rect previewRect(450, 160, 800, 450);
            cv::Rect stopButton(previewRect.x + previewRect.width / 2 - 60,
                               previewRect.y + previewRect.height + 20, 120, 40);
            if (stopButton.contains(cv::Point(x, y))) {
                isTestingCamera_ = false;
                testCapture_.release();
            }
        }
    }
}

bool CameraSelectionPage::onKey(int key) {
    // Number keys to select camera
    if (key >= '0' && key <= '9') {
        int index = key - '0';
        if (index < static_cast<int>(cameras_.size())) {
            selectedCamera_ = index;
            return true;
        }
    }
    return false;
}

bool CameraSelectionPage::isComplete() const {
    return selectedCamera_ >= 0 && selectedCamera_ < static_cast<int>(cameras_.size());
}

std::string CameraSelectionPage::getTitle() const {
    return "Camera Selection";
}

std::string CameraSelectionPage::getHelpText() const {
    return "Click on a camera to select it, or press Test to preview. Press number keys for quick selection.";
}

std::string CameraSelectionPage::validate() const {
    if (selectedCamera_ < 0) {
        return "Please select a camera";
    }
    
    // Test that camera can be opened
    cv::VideoCapture test(selectedCamera_);
    if (!test.isOpened()) {
        return "Selected camera could not be opened";
    }
    test.release();
    
    return "";
}

void CameraSelectionPage::testCamera(int index) {
    if (index < 0 || index >= static_cast<int>(cameras_.size())) {
        return;
    }
    
    if (testCapture_.isOpened()) {
        testCapture_.release();
    }
    
    testCapture_.open(index);
    if (testCapture_.isOpened()) {
        isTestingCamera_ = true;
    } else {
        std::cerr << "Failed to open camera " << index << std::endl;
        isTestingCamera_ = false;
    }
}

cv::Rect CameraSelectionPage::getCameraRect(int index) const {
    int startX = 40;
    int startY = 160;
    int cols = 5;
    
    int row = index / cols;
    int col = index % cols;
    
    int x = startX + col * (cameraBoxWidth_ + cameraBoxPadding_);
    int y = startY + row * (cameraBoxHeight_ + cameraBoxPadding_);
    
    return cv::Rect(x, y, cameraBoxWidth_, cameraBoxHeight_);
}

} // namespace pv
