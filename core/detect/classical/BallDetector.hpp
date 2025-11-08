#pragma once
#include <opencv2/opencv.hpp>
#include "../../util/Types.hpp"
#include "../../util/Config.hpp"
#include <map>

namespace pv {

class BallDetector {
public:
    struct Params { float dp=1.2f; float minDist=12; int canny=100; int acc=30; float minR=5; float maxR=30; };
    BallDetector();
    bool loadColors(const std::string &colorsYaml);
    std::vector<Ball> detect(const cv::Mat &rectified);
    Params params;
private:
    std::map<std::string, cv::Vec3f> colorPrototypes;
    std::map<std::string, int> labelMap;
};

}
