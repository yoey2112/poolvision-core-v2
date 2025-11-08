#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <sstream>

#include "../../core/util/Config.hpp"
#include "../../core/calib/Calib.hpp"
#include "../../core/detect/classical/BallDetector.hpp"
#include "../../core/detect/dl/DlDetector.hpp"
#include "../../core/track/Tracker.hpp"
#include "../../core/events/EventEngine.hpp"
#include "../../core/io/VideoSource.hpp"
#include "../../core/io/JsonSink.hpp"
#include "../../core/game/GameState.hpp"
#include "../../core/util/UiRenderer.hpp"
#include "../../core/ui/OverlayRenderer.hpp"

// Window name constant
const char* WINDOW_NAME = "Pool Vision System";

using namespace pv;

void printUsage(const char* progName) {
    std::cout << "Pool Vision System - Table Daemon\n";
    std::cout << "Usage: " << progName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --config <file>    Table configuration file (default: config/table.yaml)\n";
    std::cout << "  --camera <file>    Camera configuration file (default: config/camera.yaml)\n";
    std::cout << "  --colors <file>    Colors configuration file (default: config/colors.yaml)\n";
    std::cout << "  --source <source>  Video source: camera index (0,1,2...) or file path (default: 0)\n";
    std::cout << "  --engine <type>    Detection engine: 'classical' or 'dl' (default: classical)\n";
    std::cout << "  --fpscap <fps>     Cap frame rate (0 = unlimited, default: 0)\n";
    std::cout << "  --list-cameras     List available cameras and exit\n";
    std::cout << "  --help             Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << progName << " --source 0          # Use first camera\n";
    std::cout << "  " << progName << " --source 1          # Use second camera\n";
    std::cout << "  " << progName << " --source video.mp4  # Use video file\n";
    std::cout << "\nControls:\n";
    std::cout << "  ESC or 'q' - Quit application\n";
}

void listCameras() {
    std::cout << "Scanning for available cameras...\n";
    for(int i = 0; i < 10; i++) {
        cv::VideoCapture cap(i);
        if(cap.isOpened()) {
            double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            double fps = cap.get(cv::CAP_PROP_FPS);
            std::cout << "Camera " << i << ": " 
                      << width << "x" << height 
                      << " @ " << fps << " fps\n";
            cap.release();
        }
    }
}

int main(int argc, char** argv){
    std::string cfgTable = "config/table.yaml";
    std::string cfgCamera = "config/camera.yaml";
    std::string cfgColors = "config/colors.yaml";
    std::string source = "0";
    std::string engine = "classical";
    int fpscap = 0;

    for(int i=1;i<argc;++i){
        std::string a = argv[i];
        if(a=="--config" && i+1<argc) cfgTable = argv[++i];
        else if(a=="--camera" && i+1<argc) cfgCamera = argv[++i];
        else if(a=="--colors" && i+1<argc) cfgColors = argv[++i];
        else if(a=="--source" && i+1<argc) source = argv[++i];
        else if(a=="--engine" && i+1<argc) engine = argv[++i];
        else if(a=="--fpscap" && i+1<argc) fpscap = std::stoi(argv[++i]);
        else if(a=="--list-cameras") {
            listCameras();
            return 0;
        }
        else if(a=="--help" || a=="-h") {
            printUsage(argv[0]);
            return 0;
        }
        else {
            std::cerr << "Unknown option: " << a << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "Pool Vision System Starting...\n";
    std::cout << "  Table config: " << cfgTable << "\n";
    std::cout << "  Camera config: " << cfgCamera << "\n";
    std::cout << "  Colors config: " << cfgColors << "\n";
    std::cout << "  Video source: " << source << "\n";
    std::cout << "  Detection engine: " << engine << "\n";
    
    Calib calib; calib.load(cfgTable);
    VideoSource vs; if(!vs.open(source)){ std::cerr<<"Failed to open source "<<source<<"\n"; return 1; }
    BallDetector classical;
    classical.loadColors(cfgColors);
    DlDetector dl;
    if(engine=="dl") dl.loadModel("model.onnx");
    Tracker tracker; Config tcfg; tcfg.load(cfgTable); tracker.setTableSize({tcfg.getInt("table_width",2540), tcfg.getInt("table_height",1270)});
    EventEngine events; events.loadTable(cfgTable);
    JsonSink sink;
    
    // Initialize game state and UI
    auto gameState = std::make_shared<GameState>(GameType::EightBall);  // Default to 8-ball
    auto trackerPtr = std::make_shared<Tracker>(tracker);  // Convert to shared_ptr
    UiRenderer uiRenderer;
    OverlayRenderer overlayRenderer(gameState, trackerPtr);
    
    // Set up display window
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    cv::Mat frame;
    double ts = 0;
    while(true){
        if(!vs.read(frame)) break;
        ts += 1.0/(vs.fps()>0?vs.fps():30.0);
        // rectify
        cv::Mat rect = calib.homography.H.empty()? frame : calib.homography.warp(frame);
        std::vector<Ball> dets;
        if(engine=="dl") dets = dl.detect(rect);
        else dets = classical.detect(rect);

        // assign incremental ids for detections
        int tmpid=1;
        for(auto &d: dets) d.id = tmpid++;

        tracker.update(dets, ts);
        auto tracks = tracker.tracks();

        // Detect events
        auto pocketEvents = events.detectPocketed(dets, tracks, ts);
        
        // Convert pocket events to game events
        std::vector<Event> gameEvents;
        for (const auto& pe : pocketEvents) {
            Event e;
            e.type = EventType::Pocket;
            e.ballId = pe.ball_id;
            e.timestamp = pe.timestamp;
            gameEvents.push_back(e);
        }
        
        // Update game state
        gameState->update(tracks, gameEvents);
        
        // Create game status JSON
        std::stringstream ss;
        ss << "{";
        ss << "\"turn\":\"" << (gameState->getCurrentTurn() == PlayerTurn::Player1 ? "Player1" : "Player2") << "\",";
        ss << "\"gameOver\":" << (gameState->isGameOver() ? "true" : "false") << ",";
        if (gameState->isGameOver()) {
            ss << "\"winner\":\"" << (gameState->getWinner() == PlayerTurn::Player1 ? "Player1" : "Player2") << "\",";
        }
        ss << "\"player1Score\":" << gameState->getScore(PlayerTurn::Player1) << ",";
        ss << "\"player2Score\":" << gameState->getScore(PlayerTurn::Player2);
        ss << "}";
        
        // Prepare frame state
        FrameState state;
        state.timestamp = ts;
        state.balls = dets;
        state.tracks = tracks;
        state.events = gameEvents;
        state.gameStatus = ss.str();
        sink.emit(state);

        // Create base visualization with UI
        cv::Mat vis = uiRenderer.render(rect, *gameState, dets, tracks);
        
        // Find cue ball position
        cv::Point2f cueBallPos(-1, -1);
        for (const auto& ball : dets) {
            if (ball.id == 0) {  // Cue ball
                cueBallPos = ball.c;
                break;
            }
        }
        
        // Add real-time overlays
        cv::Mat overlay = overlayRenderer.render(vis, dets, cueBallPos);
        cv::imshow(WINDOW_NAME, overlay);
        
        char k = (char)cv::waitKey(1);
        if(k==27 || k=='q') break;
        else if(k=='t') overlayRenderer.setOverlayFlags(true, false, false, false);  // Trajectory only
        else if(k=='g') overlayRenderer.setOverlayFlags(true, true, false, false);   // Ghost ball
        else if(k=='p') overlayRenderer.setOverlayFlags(true, true, true, false);    // Position aids
        else if(k=='s') overlayRenderer.setOverlayFlags(true, true, true, true);     // All features
        else if(k=='o') overlayRenderer.setOverlayFlags(false, false, false, false); // No overlays
        
        if(fpscap>0) std::this_thread::sleep_for(std::chrono::milliseconds(1000/fpscap));
    }

    vs.release();
    return 0;
}
