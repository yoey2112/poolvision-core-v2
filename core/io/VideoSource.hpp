#pragma once
#include <opencv2/videoio.hpp>
#include <string>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>

namespace pv {

class VideoSource {
public:
    VideoSource();
    ~VideoSource();
    bool open(const std::string &source);
    bool read(cv::Mat &frame);
    double fps() const;
    void release();
private:
    void bufferFrames();
    static const int BUFFER_SIZE = 5;
    
    cv::VideoCapture cap;
    std::queue<cv::Mat> frameBuffer;
    std::mutex bufferMutex;
    std::condition_variable bufferCondition;
    std::thread bufferThread;
    bool running;
};

}
