#include "../../core/calib/Calib.hpp"
#include "../../core/io/VideoSource.hpp"
#include "../../core/util/Config.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace pv;

// Mouse callback data
struct MouseData {
    Calib *calib;
    bool addingPoint = false;
    cv::Point2f lastClick;
};

void onMouse(int event, int x, int y, int flags, void* userdata){
    MouseData* data = (MouseData*)userdata;
    if(event == cv::EVENT_LBUTTONDOWN){
        data->addingPoint = true;
        data->lastClick = cv::Point2f((float)x, (float)y);
    }
}

int main(int argc, char** argv){
    std::string tableYaml = "config/table.yaml";
    std::string source = "0";
    
    for(int i=1; i<argc; i++){
        std::string a = argv[i];
        if(a=="--config" && i+1<argc) tableYaml = argv[++i];
        else if(a=="--source" && i+1<argc) source = argv[++i];
    }
    
    // Load table config for dimensions
    Config cfg;
    if(!cfg.load(tableYaml)){
        std::cerr << "Failed to load " << tableYaml << "\n";
        return 1;
    }
    
    cv::Size tableSize(cfg.getInt("table_width", 2540), 
                      cfg.getInt("table_height", 1270));
    
    // Open video source
    VideoSource vs;
    if(!vs.open(source)){
        std::cerr << "Failed to open source " << source << "\n";
        return 1;
    }
    
    // Setup calibration
    Calib calib;
    calib.load(tableYaml); // Load existing if any
    
    cv::Mat frame;
    if(!vs.read(frame)){
        std::cerr << "Failed to read first frame\n";
        return 1;
    }
    
    calib.startCalibration(frame, tableSize);
    
    // Setup window and mouse callback
    std::string winName = "Calibration";
    cv::namedWindow(winName);
    MouseData mouseData;
    mouseData.calib = &calib;
    cv::setMouseCallback(winName, onMouse, &mouseData);
    
    std::cout << "\nCalibration Instructions:\n"
              << "1. Click points in the image\n"
              << "2. Enter corresponding table coordinates when prompted\n"
              << "3. Press 'c' to compute homography\n"
              << "4. Press 's' to save\n"
              << "5. Press ESC to exit\n\n";
    
    while(true){
        cv::Mat vis = calib.getVisualization();
        cv::imshow(winName, vis);
        
        if(mouseData.addingPoint){
            std::cout << "Enter table coordinates (x y) for clicked point: ";
            float tx, ty;
            if(std::cin >> tx >> ty){
                calib.addPoint(mouseData.lastClick, cv::Point2f(tx, ty));
                std::cout << "Added point " << calib.getPoints().size() << "\n";
            }
            mouseData.addingPoint = false;
        }
        
        char k = (char)cv::waitKey(1);
        if(k==27) break;
        else if(k=='c'){
            if(calib.computeHomography()){
                std::cout << "Homography computed successfully\n";
            } else {
                std::cout << "Failed to compute homography (need at least 4 points)\n";
            }
        }
        else if(k=='s'){
            if(calib.save(tableYaml)){
                std::cout << "Saved to " << tableYaml << "\n";
            } else {
                std::cout << "Failed to save\n";
            }
        }
        else if(k=='r'){
            calib.clearPoints();
            std::cout << "Points cleared\n";
        }
    }
    
    vs.release();
    return 0;
}