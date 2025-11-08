#pragma once
#include "../WizardPage.hpp"
#include <opencv2/opencv.hpp>

namespace pv {

enum class TableSize {
    CUSTOM,
    SIZE_7FT,
    SIZE_8FT,
    SIZE_9FT
};

class TableDimensionsPage : public WizardPage {
public:
    TableDimensionsPage();
    ~TableDimensionsPage() override = default;
    
    void init() override;
    cv::Mat render(const cv::Mat& frame, WizardConfig& config) override;
    void onMouse(int event, int x, int y, int flags) override;
    bool onKey(int key) override;
    bool isComplete() const override;
    std::string getTitle() const override;
    std::string getHelpText() const override;
    std::string validate() const override;
    
private:
    cv::Rect getSizeButtonRect(TableSize size) const;
    cv::Rect getUnitButtonRect() const;
    void applyPreset(TableSize size);
    std::string getSizeName(TableSize size) const;
    
    TableSize selectedSize_;
    double customWidth_;
    double customLength_;
    bool useMetric_;
    TableSize hoveredSize_;
};

} // namespace pv
