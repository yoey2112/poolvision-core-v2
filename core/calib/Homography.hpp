#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace pv {

class Homography {
public:
    cv::Mat H = cv::Mat::eye(3,3,CV_64F);
    bool loadFromArray(const std::vector<double>&arr);
    cv::Mat warp(const cv::Mat &frame) const;
};

}
