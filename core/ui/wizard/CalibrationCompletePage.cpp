#include "CalibrationCompletePage.hpp"

namespace pv {

CalibrationCompletePage::CalibrationCompletePage()
    : ready_(true) {
}

void CalibrationCompletePage::init() {
    ready_ = true;
}

cv::Mat CalibrationCompletePage::render(const cv::Mat& frame, WizardConfig& config) {
    cv::Mat display(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));
    
    drawTitle(display, getTitle());
    
    // Success message
    std::string message = "Calibration Complete!";
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(message, cv::FONT_HERSHEY_SIMPLEX, 1.5, 2, &baseline);
    cv::Point textPos((display.cols - textSize.width) / 2, 200);
    
    cv::putText(display, message, textPos, cv::FONT_HERSHEY_SIMPLEX, 
               1.5, cv::Scalar(100, 255, 100), 2, cv::LINE_AA);
    
    // Summary of settings
    int summaryY = 280;
    int summaryX = 200;
    
    cv::putText(display, "Calibration Summary:", cv::Point(summaryX, summaryY),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    summaryY += 50;
    
    // Camera
    std::string cameraText = "Camera: Device " + std::to_string(config.cameraIndex);
    cv::putText(display, cameraText, cv::Point(summaryX + 20, summaryY),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    summaryY += 35;
    
    // Orientation
    std::string orientText = "Rotation: " + std::to_string(config.rotation) + "\u00b0";
    if (config.flipHorizontal) orientText += ", Flipped H";
    if (config.flipVertical) orientText += ", Flipped V";
    cv::putText(display, orientText, cv::Point(summaryX + 20, summaryY),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    summaryY += 35;
    
    // Table
    std::string tableText = "Table: " + std::to_string(config.tableLength).substr(0, 4) + "m x " +
                           std::to_string(config.tableWidth).substr(0, 4) + "m";
    cv::putText(display, tableText, cv::Point(summaryX + 20, summaryY),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    summaryY += 35;
    
    // Corners
    std::string cornersText = "Corners: " + std::to_string(config.tableCorners.size()) + " marked";
    cv::putText(display, cornersText, cv::Point(summaryX + 20, summaryY),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    summaryY += 50;
    
    // Configuration files
    cv::putText(display, "Configuration will be saved to:", cv::Point(summaryX, summaryY),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    summaryY += 40;
    
    std::string files[] = {
        config.cameraConfigPath,
        config.tableConfigPath,
        config.colorsConfigPath
    };
    
    for (const auto& file : files) {
        cv::putText(display, "  - " + file, cv::Point(summaryX + 20, summaryY),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
        summaryY += 30;
    }
    
    summaryY += 30;
    
    // Instructions
    std::string instruction = "Click 'Finish' to save configuration and exit";
    cv::Size instrSize = cv::getTextSize(instruction, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseline);
    cv::Point instrPos((display.cols - instrSize.width) / 2, summaryY);
    cv::putText(display, instruction, instrPos, cv::FONT_HERSHEY_SIMPLEX,
               0.7, cv::Scalar(100, 200, 255), 1, cv::LINE_AA);
    
    // Draw checkmark icon
    int checkSize = 100;
    cv::Point checkCenter(display.cols / 2, 450);
    cv::circle(display, checkCenter, checkSize / 2, cv::Scalar(100, 255, 100), 3);
    
    // Draw checkmark
    cv::Point checkP1(checkCenter.x - 20, checkCenter.y);
    cv::Point checkP2(checkCenter.x - 5, checkCenter.y + 15);
    cv::Point checkP3(checkCenter.x + 25, checkCenter.y - 20);
    
    cv::line(display, checkP1, checkP2, cv::Scalar(100, 255, 100), 5);
    cv::line(display, checkP2, checkP3, cv::Scalar(100, 255, 100), 5);
    
    return display;
}

void CalibrationCompletePage::onMouse(int event, int x, int y, int flags) {
    // No interaction needed
}

bool CalibrationCompletePage::onKey(int key) {
    return false;
}

bool CalibrationCompletePage::isComplete() const {
    return ready_;
}

std::string CalibrationCompletePage::getTitle() const {
    return "Setup Complete";
}

std::string CalibrationCompletePage::getHelpText() const {
    return "Press 'Finish' to save and exit the wizard";
}

std::string CalibrationCompletePage::validate() const {
    return "";  // Always valid
}

} // namespace pv
