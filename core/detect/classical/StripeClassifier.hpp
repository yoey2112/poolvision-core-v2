#pragma once
#include <opencv2/opencv.hpp>

namespace pv {

class StripeClassifier {
public:
    // compute stripe score 0..1 for a candidate ball in rectified frame
    float compute(const cv::Mat &frame, const cv::Point2f &center, float radius);
};

}
