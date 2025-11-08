#include "Homography.hpp"
using namespace pv;

bool Homography::loadFromArray(const std::vector<double>&arr){
    if(arr.size()!=9) return false;
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) H.at<double>(r,c) = arr[r*3+c];
    return true;
}

cv::Mat Homography::warp(const cv::Mat &frame) const{
    // If H is identity, return original
    cv::Mat I = cv::Mat::eye(3,3,CV_64F);
    if(cv::countNonZero(H != I)==0) return frame.clone();
    cv::Mat out;
    cv::warpPerspective(frame, out, H, frame.size());
    return out;
}
