#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>

namespace pv {

class ColorLab {
public:
    static cv::Vec3f bgrToLab(const cv::Vec3b &bgr);
    static cv::Vec3f meanLab(const cv::Mat &bgr, const cv::Point &c, int radius);
    static double mahalanobis(const cv::Vec3f &x, const cv::Vec3f &mean, const cv::Matx33f &covInv);
};

} // pv
