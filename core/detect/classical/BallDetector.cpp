#include "BallDetector.hpp"
#include "StripeClassifier.hpp"
#include "../../util/ColorLab.hpp"
#include <opencv2/imgproc.hpp>
#include <limits>

using namespace pv;

BallDetector::BallDetector(){ }

bool BallDetector::loadColors(const std::string &colorsYaml){
    Config cfg;
    if(!cfg.load(colorsYaml)) return false;
    
    // Clear existing maps
    colorPrototypes.clear();
    labelMap.clear();
    
    // Process each prototype
    for(const auto &kv: cfg.arrays){
        const auto &key = kv.first;
        const auto &values = kv.second;
        if(values.size() == 3){
            colorPrototypes[key] = cv::Vec3f((float)values[0], (float)values[1], (float)values[2]);
            // Convert keys like "1", "2" etc to numeric labels
            try {
                int label = std::stoi(key);
                labelMap[key] = label;
            } catch(...){
                // Non-numeric keys like "cue" get -1
                labelMap[key] = -1;
            }
        }
    }
    return !colorPrototypes.empty();
}

std::vector<Ball> BallDetector::detect(const cv::Mat &rectified){
    std::vector<Ball> out;
    cv::Mat gray;
    
    // Convert to grayscale
    if(rectified.channels()==3) cv::cvtColor(rectified, gray, cv::COLOR_BGR2GRAY);
    else gray = rectified.clone();
    
    // Blur for circle detection
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(7,7), 2);
    
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, params.dp, params.minDist, params.canny, params.acc, params.minR, params.maxR);
    StripeClassifier sc;
    for(size_t i=0;i<circles.size();++i){
        cv::Point2f c(circles[i][0], circles[i][1]);
        float r = circles[i][2];
        // circularity: sample edge gradient consistency
        int edges = 0;
        for(int a=0;a<16;++a){
            float ang = a * 2.0f * CV_PI / 16.0f;
            int x = cv::borderInterpolate((int)std::round(c.x + r*cos(ang)), gray.cols, cv::BORDER_REFLECT);
            int y = cv::borderInterpolate((int)std::round(c.y + r*sin(ang)), gray.rows, cv::BORDER_REFLECT);
            if((int)gray.at<uchar>(y,x) < 250) edges++; // heuristic
        }
        if(edges<6) continue;
        Ball b;
        b.c = c; b.r = r;
        
        // Compute stripe score
        b.stripeScore = sc.compute(rectified, c, (int)r);
        
        // Color classification
        if(!colorPrototypes.empty()){
            cv::Vec3f ballColor = ColorLab::meanLab(rectified, c, (int)r);
            std::string bestMatch;
            double minDist = std::numeric_limits<double>::max();
            
            // Simple covariance matrix for Mahalanobis distance
            cv::Matx33f covInv = cv::Matx33f::eye();
            covInv(0,0) = 1.0f/100.0f; // L* component
            covInv(1,1) = 1.0f/50.0f;  // a* component
            covInv(2,2) = 1.0f/50.0f;  // b* component
            
            for(const auto &proto: colorPrototypes){
                double dist = ColorLab::mahalanobis(ballColor, proto.second, covInv);
                if(dist < minDist){
                    minDist = dist;
                    bestMatch = proto.first;
                }
            }
            
            // Assign label if match is good enough
            if(minDist < 10.0){ // Threshold can be tuned
                auto it = labelMap.find(bestMatch);
                if(it != labelMap.end()){
                    b.label = it->second;
                }
            }
        }
        
        out.push_back(b);
    }
    return out;
}
