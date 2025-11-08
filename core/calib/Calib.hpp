#pragma once
#include <opencv2/opencv.hpp>
#include "Homography.hpp"
#include <functional>
#include <string>

namespace pv {

class Calib {
public:
    struct CalibPoint {
        cv::Point2f imgPoint;  // Point in camera image
        cv::Point2f tablePoint; // Point in table coordinates
    };
    
    Calib();
    ~Calib();
    
    Homography homography;
    bool load(const std::string &tableYaml);
    bool save(const std::string &tableYaml) const;
    
    // Interactive calibration
    void startCalibration(const cv::Mat &img, const cv::Size &tableSize);
    void addPoint(const cv::Point2f &imgPoint, const cv::Point2f &tablePoint);
    bool computeHomography(); // Returns true if successful
    void clearPoints();
    
    // Access calibration state
    const std::vector<CalibPoint>& getPoints() const { return points; }
    cv::Mat getVisualization() const; // Get debug visualization
    
private:
    std::vector<CalibPoint> points;
    cv::Mat calibImage;
    cv::Size targetSize;
};

}
