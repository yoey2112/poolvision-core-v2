#pragma once
#include <opencv2/opencv.hpp>
#include "../../util/Types.hpp"

#ifdef USE_DL_ENGINE
#include <onnxruntime_cxx_api.h>
#endif

namespace pv {

class DlDetector {
public:
    DlDetector();
    ~DlDetector();
    bool loadModel(const std::string &path);
    std::vector<Ball> detect(const cv::Mat &rectified);
private:
    bool ready = false;
    cv::Size inputSize{640, 640};
    float confThreshold = 0.5f;
    float nmsThreshold = 0.5f;

#ifdef USE_DL_ENGINE
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<Ort::Value> processImage(const cv::Mat &img);
#endif
};

}
