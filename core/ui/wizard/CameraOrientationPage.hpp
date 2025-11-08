#pragma once
#include "../WizardPage.hpp"
#include <opencv2/opencv.hpp>

namespace pv {

class CameraOrientationPage : public WizardPage {
public:
    CameraOrientationPage();
    ~CameraOrientationPage() override = default;
    
    void init() override;
    cv::Mat render(const cv::Mat& frame, WizardConfig& config) override;
    void onMouse(int event, int x, int y, int flags) override;
    bool onKey(int key) override;
    bool isComplete() const override;
    std::string getTitle() const override;
    std::string getHelpText() const override;
    std::string validate() const override;
    
private:
    cv::Mat applyTransforms(const cv::Mat& input, int rotation, bool flipH, bool flipV);
    cv::Rect getRotationButtonRect(int rotation) const;
    cv::Rect getFlipButtonRect(const std::string& type) const;
    
    int rotation_;  // 0, 90, 180, 270
    bool flipHorizontal_;
    bool flipVertical_;
    int hoveredRotation_;
    std::string hoveredFlip_;
    bool userModified_;  // Track if user made changes
};

} // namespace pv
