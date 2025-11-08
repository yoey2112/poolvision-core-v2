#include "StripeClassifier.hpp"
#include <opencv2/imgproc.hpp>
#include <numeric>

using namespace pv;

float StripeClassifier::compute(const cv::Mat &frame, const cv::Point2f &center, float radius){
    // radial sampling: sample along rings and compute variance of intensities
    int rings = 6;
    int samples = 64;
    std::vector<float> variances;
    cv::Mat gray;
    if(frame.channels()==3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else gray = frame;
    for(int r=1;r<=rings;++r){
        float rr = radius * (0.25f + 0.75f * r / rings);
        std::vector<float> vals;
        for(int s=0;s<samples;++s){
            float a = (float)s / samples * 2.0f * CV_PI;
            int x = cv::borderInterpolate((int)std::round(center.x + rr*cos(a)), gray.cols, cv::BORDER_REFLECT);
            int y = cv::borderInterpolate((int)std::round(center.y + rr*sin(a)), gray.rows, cv::BORDER_REFLECT);
            vals.push_back((float)gray.at<uchar>(y,x));
        }
        float mean = std::accumulate(vals.begin(), vals.end(), 0.0f) / vals.size();
        float var = 0;
        for(float v: vals) var += (v-mean)*(v-mean);
        var /= vals.size();
        variances.push_back(var);
    }
    float avgVar = std::accumulate(variances.begin(), variances.end(), 0.0f) / variances.size();
    // normalize into 0..1 (empirical)
    float score = std::min(1.0f, avgVar / 200.0f);
    return score;
}
