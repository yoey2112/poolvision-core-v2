#pragma once
#include "../WizardPage.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace pv {

class TableCalibrationPage : public WizardPage {
public:
    TableCalibrationPage();
    ~TableCalibrationPage() override = default;
    
    void init() override;
    cv::Mat render(const cv::Mat& frame, WizardConfig& config) override;
    void onMouse(int event, int x, int y, int flags) override;
    bool onKey(int key) override;
    bool isComplete() const override;
    std::string getTitle() const override;
    std::string getHelpText() const override;
    std::string validate() const override;
    
private:
    void computeHomography();
    cv::Mat transformFrame(const cv::Mat& input);
    int findNearestCorner(cv::Point pos, float threshold = 20.0f);
    void drawCornerGuides(cv::Mat& img, const cv::Point& mousePos);
    void drawTransformedPreview(cv::Mat& display, const cv::Mat& frame);
    
    std::vector<cv::Point2f> corners_;  // 4 corners: TL, TR, BR, BL
    int selectedCorner_;
    int hoveredCorner_;
    bool isDragging_;
    cv::Point dragStart_;
    cv::Mat homography_;
    bool homographyValid_;
    cv::Rect previewRect_;
    
    const std::vector<std::string> cornerNames_ = {
        "Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"
    };
};

} // namespace pv
