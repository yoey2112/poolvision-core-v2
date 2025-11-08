#include "TableDimensionsPage.hpp"

namespace pv {

TableDimensionsPage::TableDimensionsPage()
    : selectedSize_(TableSize::SIZE_9FT), customWidth_(4.57), customLength_(2.54),
      useMetric_(true), hoveredSize_(TableSize::CUSTOM) {
}

void TableDimensionsPage::init() {
    selectedSize_ = TableSize::SIZE_9FT;
    applyPreset(selectedSize_);
    useMetric_ = true;
}

cv::Mat TableDimensionsPage::render(const cv::Mat& frame, WizardConfig& config) {
    cv::Mat display(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));
    
    drawTitle(display, getTitle());
    
    // Instructions
    std::string instructions = "Select your table size or enter custom dimensions:";
    cv::putText(display, instructions, cv::Point(40, 120),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    
    // Preset buttons
    int buttonY = 180;
    int buttonX = 100;
    int buttonWidth = 200;
    int buttonHeight = 80;
    int spacing = 40;
    
    TableSize sizes[] = {TableSize::SIZE_7FT, TableSize::SIZE_8FT, TableSize::SIZE_9FT, TableSize::CUSTOM};
    for (int i = 0; i < 4; i++) {
        cv::Rect btnRect = getSizeButtonRect(sizes[i]);
        bool isSelected = (selectedSize_ == sizes[i]);
        bool isHovered = (hoveredSize_ == sizes[i]);
        
        std::string label = getSizeName(sizes[i]);
        drawButton(display, label, btnRect, isHovered || isSelected, true);
        
        // Draw dimensions below button
        if (sizes[i] != TableSize::CUSTOM) {
            TableSize temp = selectedSize_;
            selectedSize_ = sizes[i];
            applyPreset(selectedSize_);
            selectedSize_ = temp;
            
            std::string dims;
            if (useMetric_) {
                dims = std::to_string(customLength_).substr(0, 4) + "m x " +
                       std::to_string(customWidth_).substr(0, 4) + "m";
            } else {
                dims = std::to_string(static_cast<int>(customLength_ * 3.28084)) + "' x " +
                       std::to_string(static_cast<int>(customWidth_ * 3.28084)) + "'";
            }
            
            cv::putText(display, dims, cv::Point(btnRect.x + 20, btnRect.y + btnRect.height + 25),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
        }
        
        if (isSelected) {
            cv::circle(display, cv::Point(btnRect.x + 20, btnRect.y + btnRect.height / 2),
                      8, cv::Scalar(100, 255, 100), -1);
        }
    }
    
    // Unit toggle
    buttonY = 350;
    cv::Rect unitRect = getUnitButtonRect();
    std::string unitLabel = useMetric_ ? "Metric (meters)" : "Imperial (feet)";
    drawButton(display, unitLabel, unitRect, false, true);
    
    // Current dimensions display
    int displayY = 450;
    cv::putText(display, "Selected Dimensions:", cv::Point(100, displayY),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    displayY += 40;
    
    std::string widthStr, lengthStr;
    if (useMetric_) {
        widthStr = "Width: " + std::to_string(customWidth_).substr(0, 5) + " meters";
        lengthStr = "Length: " + std::to_string(customLength_).substr(0, 5) + " meters";
    } else {
        double widthFt = customWidth_ * 3.28084;
        double lengthFt = customLength_ * 3.28084;
        widthStr = "Width: " + std::to_string(widthFt).substr(0, 5) + " feet";
        lengthStr = "Length: " + std::to_string(lengthFt).substr(0, 5) + " feet";
    }
    
    cv::putText(display, widthStr, cv::Point(120, displayY),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    displayY += 30;
    cv::putText(display, lengthStr, cv::Point(120, displayY),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    
    // Diagram
    int diagramX = 700;
    int diagramY = 200;
    int diagramWidth = 400;
    int diagramHeight = 250;
    
    cv::rectangle(display, cv::Rect(diagramX, diagramY, diagramWidth, diagramHeight),
                 cv::Scalar(100, 200, 255), 2);
    cv::rectangle(display, cv::Rect(diagramX, diagramY, diagramWidth, diagramHeight),
                 cv::Scalar(100, 200, 255, 50), -1);
    
    // Labels on diagram
    std::string lengthLabel = useMetric_ ? 
        std::to_string(customLength_).substr(0, 4) + "m" :
        std::to_string(static_cast<int>(customLength_ * 3.28084)) + "ft";
    std::string widthLabel = useMetric_ ?
        std::to_string(customWidth_).substr(0, 4) + "m" :
        std::to_string(static_cast<int>(customWidth_ * 3.28084)) + "ft";
    
    cv::putText(display, lengthLabel, cv::Point(diagramX + diagramWidth / 2 - 30, diagramY - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    
    cv::putText(display, widthLabel, cv::Point(diagramX - 80, diagramY + diagramHeight / 2),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    
    // Update config
    config.tableWidth = customWidth_;
    config.tableLength = customLength_;
    config.useMetric = useMetric_;
    
    return display;
}

void TableDimensionsPage::onMouse(int event, int x, int y, int flags) {
    if (event == cv::EVENT_MOUSEMOVE) {
        hoveredSize_ = TableSize::CUSTOM;
        
        TableSize sizes[] = {TableSize::SIZE_7FT, TableSize::SIZE_8FT, TableSize::SIZE_9FT, TableSize::CUSTOM};
        for (int i = 0; i < 4; i++) {
            cv::Rect rect = getSizeButtonRect(sizes[i]);
            if (rect.contains(cv::Point(x, y))) {
                hoveredSize_ = sizes[i];
                break;
            }
        }
    }
    else if (event == cv::EVENT_LBUTTONDOWN) {
        // Check size buttons
        TableSize sizes[] = {TableSize::SIZE_7FT, TableSize::SIZE_8FT, TableSize::SIZE_9FT, TableSize::CUSTOM};
        for (int i = 0; i < 4; i++) {
            cv::Rect rect = getSizeButtonRect(sizes[i]);
            if (rect.contains(cv::Point(x, y))) {
                selectedSize_ = sizes[i];
                applyPreset(selectedSize_);
                return;
            }
        }
        
        // Check unit button
        cv::Rect unitRect = getUnitButtonRect();
        if (unitRect.contains(cv::Point(x, y))) {
            useMetric_ = !useMetric_;
        }
    }
}

bool TableDimensionsPage::onKey(int key) {
    if (key == 'u' || key == 'U') {
        useMetric_ = !useMetric_;
        return true;
    }
    
    // Number keys for presets
    if (key >= '7' && key <= '9') {
        int size = key - '0';
        if (size == 7) selectedSize_ = TableSize::SIZE_7FT;
        else if (size == 8) selectedSize_ = TableSize::SIZE_8FT;
        else if (size == 9) selectedSize_ = TableSize::SIZE_9FT;
        applyPreset(selectedSize_);
        return true;
    }
    
    return false;
}

bool TableDimensionsPage::isComplete() const {
    return true;  // Always complete - has default value
}

std::string TableDimensionsPage::getTitle() const {
    return "Table Dimensions";
}

std::string TableDimensionsPage::getHelpText() const {
    return "Press 7/8/9 for standard sizes, U to toggle units. Click buttons to select.";
}

std::string TableDimensionsPage::validate() const {
    if (customWidth_ <= 0 || customLength_ <= 0) {
        return "Invalid table dimensions";
    }
    return "";
}

cv::Rect TableDimensionsPage::getSizeButtonRect(TableSize size) const {
    int buttonX = 100;
    int buttonY = 180;
    int buttonWidth = 200;
    int buttonHeight = 80;
    int spacing = 40;
    
    int index = 0;
    if (size == TableSize::SIZE_7FT) index = 0;
    else if (size == TableSize::SIZE_8FT) index = 1;
    else if (size == TableSize::SIZE_9FT) index = 2;
    else if (size == TableSize::CUSTOM) index = 3;
    
    int x = buttonX + (index % 2) * (buttonWidth + spacing);
    int y = buttonY + (index / 2) * (buttonHeight + spacing);
    
    return cv::Rect(x, y, buttonWidth, buttonHeight);
}

cv::Rect TableDimensionsPage::getUnitButtonRect() const {
    return cv::Rect(100, 350, 200, 50);
}

void TableDimensionsPage::applyPreset(TableSize size) {
    switch (size) {
        case TableSize::SIZE_7FT:
            customLength_ = 3.9624;  // 7 feet in meters (length)
            customWidth_ = 1.9812;   // 3.5 feet (width)
            break;
        case TableSize::SIZE_8FT:
            customLength_ = 4.4196;  // 8 feet
            customWidth_ = 2.2098;   // 4 feet
            break;
        case TableSize::SIZE_9FT:
            customLength_ = 4.57;    // 9 feet (tournament size)
            customWidth_ = 2.54;     // 4.5 feet
            break;
        case TableSize::CUSTOM:
            // Keep existing values
            break;
    }
}

std::string TableDimensionsPage::getSizeName(TableSize size) const {
    switch (size) {
        case TableSize::SIZE_7FT: return "7-Foot Table";
        case TableSize::SIZE_8FT: return "8-Foot Table";
        case TableSize::SIZE_9FT: return "9-Foot Table";
        case TableSize::CUSTOM: return "Custom Size";
        default: return "Unknown";
    }
}

} // namespace pv
