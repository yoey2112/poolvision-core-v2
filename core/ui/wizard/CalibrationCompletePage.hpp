#pragma once
#include "../WizardPage.hpp"
#include <opencv2/opencv.hpp>

namespace pv {

class CalibrationCompletePage : public WizardPage {
public:
    CalibrationCompletePage();
    ~CalibrationCompletePage() override = default;
    
    void init() override;
    cv::Mat render(const cv::Mat& frame, WizardConfig& config) override;
    void onMouse(int event, int x, int y, int flags) override;
    bool onKey(int key) override;
    bool isComplete() const override;
    std::string getTitle() const override;
    std::string getHelpText() const override;
    std::string validate() const override;
    
private:
    bool ready_;
};

} // namespace pv
