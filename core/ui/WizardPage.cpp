#include "WizardPage.hpp"

namespace pv {

void WizardPage::drawButton(cv::Mat& img, const std::string& text, cv::Rect rect, 
                           bool highlighted, bool enabled) {
    cv::Scalar bgColor = highlighted ? cv::Scalar(100, 200, 255) : cv::Scalar(80, 80, 80);
    cv::Scalar textColor = enabled ? cv::Scalar(255, 255, 255) : cv::Scalar(128, 128, 128);
    
    if (!enabled) {
        bgColor = cv::Scalar(50, 50, 50);
    }
    
    // Draw button background
    cv::rectangle(img, rect, bgColor, -1);
    cv::rectangle(img, rect, highlighted ? cv::Scalar(150, 220, 255) : cv::Scalar(100, 100, 100), 2);
    
    // Draw text centered
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
    cv::Point textPos(rect.x + (rect.width - textSize.width) / 2,
                     rect.y + (rect.height + textSize.height) / 2);
    cv::putText(img, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 1, cv::LINE_AA);
}

void WizardPage::drawTextBox(cv::Mat& img, const std::string& text, cv::Rect rect) {
    cv::rectangle(img, rect, cv::Scalar(40, 40, 40), -1);
    cv::rectangle(img, rect, cv::Scalar(100, 100, 100), 1);
    
    cv::Point textPos(rect.x + 10, rect.y + rect.height / 2 + 5);
    cv::putText(img, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, 
               cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

void WizardPage::drawSlider(cv::Mat& img, const std::string& label, cv::Rect rect,
                           float value, float min, float max) {
    // Draw label
    cv::putText(img, label, cv::Point(rect.x, rect.y - 5), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    
    // Draw slider track
    cv::Rect track(rect.x, rect.y + rect.height / 2 - 2, rect.width, 4);
    cv::rectangle(img, track, cv::Scalar(80, 80, 80), -1);
    
    // Draw slider thumb
    float normalizedValue = (value - min) / (max - min);
    int thumbX = rect.x + static_cast<int>(normalizedValue * rect.width);
    cv::circle(img, cv::Point(thumbX, rect.y + rect.height / 2), 8, 
              cv::Scalar(100, 200, 255), -1);
    cv::circle(img, cv::Point(thumbX, rect.y + rect.height / 2), 8, 
              cv::Scalar(150, 220, 255), 2);
    
    // Draw value
    std::string valueText = std::to_string(static_cast<int>(value));
    cv::putText(img, valueText, cv::Point(rect.x + rect.width + 10, rect.y + rect.height / 2 + 5),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

void WizardPage::drawCheckbox(cv::Mat& img, const std::string& label, cv::Point pos,
                             bool checked) {
    cv::Rect box(pos.x, pos.y, 20, 20);
    cv::rectangle(img, box, cv::Scalar(100, 100, 100), 2);
    
    if (checked) {
        cv::line(img, cv::Point(pos.x + 4, pos.y + 10),
                cv::Point(pos.x + 8, pos.y + 16), cv::Scalar(100, 200, 255), 2);
        cv::line(img, cv::Point(pos.x + 8, pos.y + 16),
                cv::Point(pos.x + 16, pos.y + 4), cv::Scalar(100, 200, 255), 2);
    }
    
    cv::putText(img, label, cv::Point(pos.x + 30, pos.y + 15),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

void WizardPage::drawTitle(cv::Mat& img, const std::string& title) {
    cv::Rect header(0, 0, img.cols, 80);
    cv::rectangle(img, header, cv::Scalar(30, 30, 30), -1);
    
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 1.2, 2, &baseline);
    cv::Point textPos((img.cols - textSize.width) / 2, 50);
    
    // Shadow
    cv::putText(img, title, textPos + cv::Point(2, 2), cv::FONT_HERSHEY_SIMPLEX, 
               1.2, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    // Text
    cv::putText(img, title, textPos, cv::FONT_HERSHEY_SIMPLEX, 
               1.2, cv::Scalar(100, 200, 255), 2, cv::LINE_AA);
}

void WizardPage::drawHelpBar(cv::Mat& img, const std::string& help) {
    cv::Rect footer(0, img.rows - 40, img.cols, 40);
    cv::rectangle(img, footer, cv::Scalar(30, 30, 30), -1);
    
    cv::putText(img, help, cv::Point(20, img.rows - 15),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
}

void WizardPage::drawProgressBar(cv::Mat& img, int current, int total) {
    int barWidth = 300;
    int barHeight = 10;
    cv::Point barPos(img.cols - barWidth - 20, 35);
    
    // Background
    cv::rectangle(img, cv::Rect(barPos.x, barPos.y, barWidth, barHeight),
                 cv::Scalar(50, 50, 50), -1);
    
    // Progress
    int progressWidth = (barWidth * current) / total;
    cv::rectangle(img, cv::Rect(barPos.x, barPos.y, progressWidth, barHeight),
                 cv::Scalar(100, 200, 255), -1);
    
    // Text
    std::string progressText = std::to_string(current) + " / " + std::to_string(total);
    cv::putText(img, progressText, cv::Point(barPos.x + barWidth + 10, barPos.y + barHeight),
               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
}

} // namespace pv
