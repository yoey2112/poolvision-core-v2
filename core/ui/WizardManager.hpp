#pragma once
#include "WizardPage.hpp"
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

namespace pv {

/**
 * @brief Manages wizard flow and page navigation
 */
class WizardManager {
public:
    WizardManager();
    ~WizardManager() = default;
    
    /**
     * @brief Add a page to the wizard
     */
    void addPage(std::unique_ptr<WizardPage> page);
    
    /**
     * @brief Run the wizard
     * @return true if wizard completed successfully, false if cancelled
     */
    bool run();
    
    /**
     * @brief Get the final configuration
     */
    const WizardConfig& getConfig() const { return config_; }
    
    /**
     * @brief Save configuration to files
     */
    bool saveConfig();
    
private:
    void nextPage();
    void previousPage();
    void renderCurrentPage(const cv::Mat& frame);
    void handleMouse(int event, int x, int y, int flags);
    void handleKeyboard(int key);
    
    static void onMouse(int event, int x, int y, int flags, void* userdata);
    
    std::vector<std::unique_ptr<WizardPage>> pages_;
    size_t currentPage_;
    WizardConfig config_;
    cv::VideoCapture camera_;
    cv::Mat displayImage_;
    bool running_;
    bool completed_;
    
    const std::string windowName_ = "Pool Vision Setup Wizard";
};

} // namespace pv
