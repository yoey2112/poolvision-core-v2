#pragma once
#include "../WizardPage.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace pv {

struct CameraInfo {
    int index;
    std::string name;
    cv::Size resolution;
    bool available;
    cv::Mat thumbnail;
};

class CameraSelectionPage : public WizardPage {
public:
    CameraSelectionPage();
    ~CameraSelectionPage() override = default;
    
    void init() override;
    cv::Mat render(const cv::Mat& frame, WizardConfig& config) override;
    void onMouse(int event, int x, int y, int flags) override;
    bool onKey(int key) override;
    bool isComplete() const override;
    std::string getTitle() const override;
    std::string getHelpText() const override;
    std::string validate() const override;
    
private:
    void enumerateCameras();
    void testCamera(int index);
    void captureThumbnail(int index);
    cv::Rect getCameraRect(int index) const;
    
    std::vector<CameraInfo> cameras_;
    int selectedCamera_;
    int hoveredCamera_;
    bool isTestingCamera_;
    cv::VideoCapture testCapture_;
    cv::Mat testFrame_;
    
    const int cameraBoxWidth_ = 200;
    const int cameraBoxHeight_ = 180;
    const int cameraBoxPadding_ = 20;
};

} // namespace pv
