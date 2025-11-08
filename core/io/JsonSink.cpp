#include "JsonSink.hpp"
#include <iostream>
#include <sstream>

using namespace pv;

void JsonSink::emit(const FrameState &s){
    std::ostringstream ss;
    ss << "{\"timestamp\":" << s.timestamp << ",\"balls\": [";
    for(size_t i=0;i<s.balls.size();++i){
        auto &b = s.balls[i];
        ss << "{\"id\":"<<b.id<<",\"x\":"<<b.c.x<<",\"y\":"<<b.c.y<<",\"r\":"<<b.r<<",\"stripe\":"<<b.stripeScore<<",\"label\":"<<b.label<<"}";
        if(i+1<s.balls.size()) ss << ",";
    }
    ss << "],\"tracks\": [";
    for(size_t i=0;i<s.tracks.size();++i){
        auto &t = s.tracks[i];
        ss << "{\"id\":"<<t.id<<",\"x\":"<<t.c.x<<",\"y\":"<<t.c.y<<",\"vx\":"<<t.v.x<<",\"vy\":"<<t.v.y<<"}";
        if(i+1<s.tracks.size()) ss << ",";
    }
    ss << "],\"events\": [";
    for(size_t i=0;i<s.events.size();++i){
        auto &e = s.events[i];
        ss << "{\"type\":"<<static_cast<int>(e.type)<<",\"ball_id\":"<<e.ballId<<",\"t\":"<<e.timestamp<<"}";
        if(i+1<s.events.size()) ss << ",";
    }
    ss << "]}";
    std::cout << ss.str() << std::endl;
}
