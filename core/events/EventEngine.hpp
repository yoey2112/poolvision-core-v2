#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "../util/Types.hpp"
#include <map>

namespace pv {

class EventEngine {
public:
    bool loadTable(const std::string &tableYaml);
    std::vector<PocketEvent> detectPocketed(const std::vector<Ball>&balls, const std::vector<Track>&tracks, double timestamp);
private:
    std::vector<std::vector<cv::Point>> pockets;
    cv::Size tableSize{2540, 1270};
    float predictTimeWindow = 0.5f; // seconds to look ahead
    
    bool isPocketed(const cv::Point2f &pos, int &pocketIdx) const;
    bool willReachPocket(const cv::Point2f &pos, const cv::Point2f &vel, float radius, 
                        int &pocketIdx, double &timeToHit) const;
    std::map<int, double> lastPocketTime; // track ball IDs that were recently pocketed
};

}
