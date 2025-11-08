#include "EventEngine.hpp"
#include "../util/Config.hpp"
#include <iostream>

using namespace pv;

bool EventEngine::loadTable(const std::string &tableYaml){
    Config cfg;
    if(!cfg.load(tableYaml)) return false;
    auto arr = cfg.getArray("pockets");
    
    tableSize.width = cfg.getInt("table_width", 2540);
    tableSize.height = cfg.getInt("table_height", 1270);
    
    pockets.clear();
    pockets.resize(6);
    pockets[0] = { {50,50},{120,50},{120,120},{50,120} };
    pockets[1] = { {tableSize.width/2-20,50},{tableSize.width/2+80,50},
                   {tableSize.width/2+80,120},{tableSize.width/2-20,120} };
    pockets[2] = { {tableSize.width-170,50},{tableSize.width-100,50},
                   {tableSize.width-100,120},{tableSize.width-170,120} };
    pockets[3] = { {50,tableSize.height-170},{120,tableSize.height-170},
                   {120,tableSize.height-100},{50,tableSize.height-100} };
    pockets[4] = { {tableSize.width/2-20,tableSize.height-170},{tableSize.width/2+80,tableSize.height-170},
                   {tableSize.width/2+80,tableSize.height-100},{tableSize.width/2-20,tableSize.height-100} };
    pockets[5] = { {tableSize.width-170,tableSize.height-170},{tableSize.width-100,tableSize.height-170},
                   {tableSize.width-100,tableSize.height-100},{tableSize.width-170,tableSize.height-100} };
    return true;
}

bool EventEngine::isPocketed(const cv::Point2f &pos, int &pocketIdx) const {
    cv::Point pt(cv::saturate_cast<int>(pos.x), cv::saturate_cast<int>(pos.y));
    for(size_t i=0; i<pockets.size(); ++i){
        if(cv::pointPolygonTest(pockets[i], pt, false) >= 0){
            pocketIdx = (int)i;
            return true;
        }
    }
    return false;
}

bool EventEngine::willReachPocket(const cv::Point2f &pos, const cv::Point2f &vel, 
                                float radius, int &pocketIdx, double &timeToHit) const {
    // Simple linear trajectory prediction
    float speed = cv::norm(vel);
    if(speed < 1.0f) return false; // Too slow
    
    cv::Point2f dir = vel / speed;
    float maxDist = speed * predictTimeWindow;
    cv::Point2f futurePos = pos + maxDist * dir;
    
    // Check for cushion collisions
    if(futurePos.x < radius || futurePos.x > tableSize.width - radius ||
       futurePos.y < radius || futurePos.y > tableSize.height - radius) {
        return false; // Simplified - ignoring rebounds
    }
    
    // Check each pocket
    for(size_t i=0; i<pockets.size(); ++i){
        // Use pocket center as approximation
        cv::Point2f pocketCenter(0,0);
        for(const auto &p: pockets[i]) pocketCenter += cv::Point2f(p.x, p.y);
        pocketCenter *= 1.0f/pockets[i].size();
        
        // Vector from ball to pocket
        cv::Point2f toPocket = pocketCenter - pos;
        float distToPocket = cv::norm(toPocket);
        
        // Check if ball is heading towards this pocket
        float cosAngle = dir.dot(toPocket/distToPocket);
        if(cosAngle > 0.866f){ // Within ~30 degrees
            // Project velocity onto direction to pocket
            float approachSpeed = speed * cosAngle;
            timeToHit = distToPocket / approachSpeed;
            
            if(timeToHit <= predictTimeWindow){
                pocketIdx = (int)i;
                return true;
            }
        }
    }
    return false;
}

std::vector<PocketEvent> EventEngine::detectPocketed(const std::vector<Ball>&balls, 
                                                   const std::vector<Track>&tracks, 
                                                   double timestamp){
    std::vector<PocketEvent> ev;
    
    // Clean up old entries from lastPocketTime
    for(auto it = lastPocketTime.begin(); it != lastPocketTime.end();){
        if(timestamp - it->second > 2.0){ // Remove after 2 seconds
            it = lastPocketTime.erase(it);
        } else {
            ++it;
        }
    }
    
    // Check immediate pockets
    for(const auto &b: balls){
        // Skip if recently pocketed
        if(lastPocketTime.find(b.id) != lastPocketTime.end()) continue;
        
        int pocketIdx;
        if(isPocketed(b.c, pocketIdx)){
            PocketEvent pe;
            pe.ball_id = b.id;
            pe.pocket_idx = pocketIdx;
            pe.timestamp = timestamp;
            ev.push_back(pe);
            lastPocketTime[b.id] = timestamp;
        }
    }
    
    // Predict future pockets based on tracked velocities
    for(const auto &t: tracks){
        // Skip if recently pocketed or already detected
        if(lastPocketTime.find(t.id) != lastPocketTime.end()) continue;
        
        int pocketIdx;
        double timeToHit;
        if(willReachPocket(t.c, t.v, t.r, pocketIdx, timeToHit)){
            PocketEvent pe;
            pe.ball_id = t.id;
            pe.pocket_idx = pocketIdx;
            pe.timestamp = timestamp + timeToHit; // Predicted future time
            ev.push_back(pe);
            // Don't add to lastPocketTime yet - only when actually pocketed
        }
    }
    
    return ev;
}
