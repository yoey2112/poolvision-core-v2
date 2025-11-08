#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <functional>
#include <map>

namespace pv {

/**
 * @brief Configuration data structure for wizard pages to read/write
 */
struct WizardConfig {
    // Camera settings
    int cameraIndex = 0;
    int rotation = 0;  // 0, 90, 180, 270
    bool flipHorizontal = false;
    bool flipVertical = false;
    cv::Size resolution = cv::Size(1920, 1080);
    
    // Table calibration
    std::vector<cv::Point2f> tableCorners;  // 4 corners in image space
    cv::Mat homographyMatrix;
    
    // Table dimensions
    double tableWidth = 2.54;   // meters (8ft standard)
    double tableLength = 4.57;  // meters (9ft standard)
    bool useMetric = true;
    
    // Pockets
    std::vector<cv::Point2f> pocketPositions;  // 6 pockets in world space
    std::vector<float> pocketRadii;            // radius for each pocket
    
    // Ball colors (LAB color space)
    std::map<int, cv::Vec3f> ballColors;  // ball number -> LAB color
    float colorTolerance = 20.0f;
    
    // Detection parameters
    int minRadius = 10;
    int maxRadius = 50;
    double detectionSensitivity = 0.8;
    
    // Save paths
    std::string cameraConfigPath = "config/camera.yaml";
    std::string tableConfigPath = "config/table.yaml";
    std::string colorsConfigPath = "config/colors.yaml";
};

/**
 * @brief Abstract base class for wizard pages
 */
class WizardPage {
public:
    virtual ~WizardPage() = default;
    
    /**
     * @brief Initialize the page (called when entering)
     */
    virtual void init() = 0;
    
    /**
     * @brief Update and render the page
     * @param frame Current camera frame (if applicable)
     * @param config Shared configuration data
     * @return Output image to display
     */
    virtual cv::Mat render(const cv::Mat& frame, WizardConfig& config) = 0;
    
    /**
     * @brief Handle mouse events
     */
    virtual void onMouse(int event, int x, int y, int flags) = 0;
    
    /**
     * @brief Handle keyboard events
     * @param key Pressed key code
     * @return true if page handled the key
     */
    virtual bool onKey(int key) = 0;
    
    /**
     * @brief Check if page is complete and can proceed
     */
    virtual bool isComplete() const = 0;
    
    /**
     * @brief Get page title
     */
    virtual std::string getTitle() const = 0;
    
    /**
     * @brief Get help text for current page
     */
    virtual std::string getHelpText() const = 0;
    
    /**
     * @brief Validate page data before proceeding
     * @return Empty string if valid, error message otherwise
     */
    virtual std::string validate() const = 0;
    
    // Helper methods to draw styled UI elements (public for use by WizardManager)
    void drawButton(cv::Mat& img, const std::string& text, cv::Rect rect, 
                   bool highlighted = false, bool enabled = true);
    void drawTextBox(cv::Mat& img, const std::string& text, cv::Rect rect);
    void drawSlider(cv::Mat& img, const std::string& label, cv::Rect rect,
                   float value, float min, float max);
    void drawCheckbox(cv::Mat& img, const std::string& label, cv::Point pos,
                     bool checked);
    void drawTitle(cv::Mat& img, const std::string& title);
    void drawHelpBar(cv::Mat& img, const std::string& help);
    void drawProgressBar(cv::Mat& img, int current, int total);
};

} // namespace pv
