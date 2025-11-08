#include "ColorLab.hpp"
#include <opencv2/imgproc.hpp>

using namespace pv;

cv::Vec3f ColorLab::bgrToLab(const cv::Vec3b &bgr){
    cv::Mat3b bgrMat(1,1);
    bgrMat(0,0)=bgr;
    cv::Mat3f lab;
    bgrMat.convertTo(bgrMat, CV_8UC3);
    cv::cvtColor(bgrMat, lab, cv::COLOR_BGR2Lab);
    cv::Vec3f v = lab.at<cv::Vec3f>(0,0);
    return v;
}

cv::Vec3f ColorLab::meanLab(const cv::Mat &bgr, const cv::Point &c, int radius){
    cv::Rect r(c.x-radius, c.y-radius, radius*2+1, radius*2+1);
    r &= cv::Rect(0,0,bgr.cols,bgr.rows);
    cv::Mat roi = bgr(r);
    cv::Mat lab;
    cv::cvtColor(roi, lab, cv::COLOR_BGR2Lab);
    cv::Scalar m = cv::mean(lab);
    return cv::Vec3f((float)m[0], (float)m[1], (float)m[2]);
}

double ColorLab::mahalanobis(const cv::Vec3f &x, const cv::Vec3f &mean, const cv::Matx33f &covInv){
    cv::Vec3f d = x - mean;
    cv::Matx31f dv(d[0],d[1],d[2]);
    cv::Matx13f dvT = dv.t();
    float res = (float)(dvT * covInv * dv)(0,0);
    return res;
}
