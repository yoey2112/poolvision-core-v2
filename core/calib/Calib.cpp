#include "Calib.hpp"
#include <fstream>
#include "../util/Config.hpp"
#include <sstream>

using namespace pv;

Calib::Calib() {}
Calib::~Calib() {}

bool Calib::load(const std::string &tableYaml){
    Config cfg;
    if(!cfg.load(tableYaml)) return false;
    auto h = cfg.getArray("homography");
    if(h.size()==9) homography.loadFromArray(h);
    return true;
}

bool Calib::save(const std::string &tableYaml) const {
    std::ofstream out(tableYaml);
    if(!out) return false;
    
    // Preserve existing content except homography
    Config cfg;
    cfg.load(tableYaml);
    
    // Update homography
    std::vector<double> h(9);
    for(int i=0; i<9; i++) h[i] = homography.H.at<double>(i/3,i%3);
    cfg.arrays["homography"] = h;
    
    // Write back
    for(const auto &kv: cfg.kv){
        out << kv.first << ": " << kv.second << "\n";
    }
    for(const auto &arr: cfg.arrays){
        out << arr.first << ": [";
        for(size_t i=0; i<arr.second.size(); i++){
            out << arr.second[i];
            if(i+1<arr.second.size()) out << ", ";
        }
        out << "]\n";
    }
    
    return true;
}

void Calib::startCalibration(const cv::Mat &img, const cv::Size &tableSize){
    calibImage = img.clone();
    targetSize = tableSize;
    points.clear();
}

void Calib::addPoint(const cv::Point2f &imgPoint, const cv::Point2f &tablePoint){
    CalibPoint cp;
    cp.imgPoint = imgPoint;
    cp.tablePoint = tablePoint;
    points.push_back(cp);
}

bool Calib::computeHomography(){
    if(points.size() < 4) return false;
    
    std::vector<cv::Point2f> src, dst;
    for(const auto &p: points){
        src.push_back(p.imgPoint);
        dst.push_back(p.tablePoint);
    }
    
    // Compute homography using RANSAC
    homography.H = cv::findHomography(src, dst, cv::RANSAC);
    return !homography.H.empty();
}

void Calib::clearPoints(){
    points.clear();
}

cv::Mat Calib::getVisualization() const {
    cv::Mat vis;
    if(calibImage.empty()) return vis;
    
    cv::cvtColor(calibImage, vis, cv::COLOR_BGR2BGRA);
    
    // Draw table outline
    std::vector<cv::Point2f> tableCorners = {
        {0, 0},
        {(float)targetSize.width, 0},
        {(float)targetSize.width, (float)targetSize.height},
        {0, (float)targetSize.height}
    };
    
    if(!homography.H.empty()){
        // Transform table corners to image space
        cv::perspectiveTransform(tableCorners, tableCorners, homography.H.inv());
        // Draw table outline
        for(int i=0; i<4; i++){
            cv::line(vis, tableCorners[i], tableCorners[(i+1)%4], cv::Scalar(0,255,0,128), 2);
        }
    }
    
    // Draw calibration points
    for(size_t i=0; i<points.size(); i++){
        cv::circle(vis, points[i].imgPoint, 5, cv::Scalar(0,0,255,255), -1);
        cv::putText(vis, std::to_string(i+1), points[i].imgPoint + cv::Point2f(10,10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255,255), 2);
    }
    
    return vis;
}
