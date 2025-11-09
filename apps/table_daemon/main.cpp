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
#include "../../core/track/modern/ByteTrackMOT.hpp"
#include "../../core/performance/ProcessingIsolation.hpp"
#include "../../core/events/EventEngine.hpp"
#include "../../core/io/VideoSource.hpp"
#include "../../core/io/JsonSink.hpp"
#include "../../core/game/GameState.hpp"
#include "../../core/game/ModernGameLogicAdapter.hpp"
#include "../../core/util/UiRenderer.hpp"
#include "../../core/ui/OverlayRenderer.hpp"

#ifdef USE_OLLAMA
#include "../../core/ai/CoachingEngine.hpp"
#endif

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
    std::cout << "  --tracker <type>   Tracking engine: 'legacy' or 'bytetrack' (default: legacy)\n";
    std::cout << "  --gamelogic <type> Game logic: 'legacy' or 'modern' (default: legacy)\n";
    std::cout << "  --fpscap <fps>     Cap frame rate (0 = unlimited, default: 0)\n";
#ifdef USE_OLLAMA
    std::cout << "  --coaching         Enable AI coaching system (requires Ollama)\n";
    std::cout << "  --coach-personality <type> Coaching personality: 'supportive', 'analytical', 'challenging' (default: supportive)\n";
#endif
    std::cout << "  --list-cameras     List available cameras and exit\n";
    std::cout << "  --help             Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << progName << " --source 0          # Use first camera\n";
    std::cout << "  " << progName << " --source 1          # Use second camera\n";
    std::cout << "  " << progName << " --source video.mp4  # Use video file\n";
    std::cout << "Controls:\n";
    std::cout << "  ESC or 'q' - Quit application\n";
    std::cout << "  't' - Show trajectory overlay only\n";
    std::cout << "  'g' - Show trajectory and ghost ball\n";
    std::cout << "  'p' - Show position aids\n";
    std::cout << "  's' - Show all overlays\n";
    std::cout << "  'o' - Hide all overlays\n";
#ifdef USE_OLLAMA
    std::cout << "  'c' - Request coaching advice (when coaching enabled)\n";
#endif
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
    std::string trackerType = "legacy";
    std::string gameLogicType = "legacy";
    int fpscap = 0;
#ifdef USE_OLLAMA
    bool enableCoaching = false;
    std::string coachingPersonality = "supportive";
#endif

    for(int i=1;i<argc;++i){
        std::string a = argv[i];
        if(a=="--config" && i+1<argc) cfgTable = argv[++i];
        else if(a=="--camera" && i+1<argc) cfgCamera = argv[++i];
        else if(a=="--colors" && i+1<argc) cfgColors = argv[++i];
        else if(a=="--source" && i+1<argc) source = argv[++i];
        else if(a=="--engine" && i+1<argc) engine = argv[++i];
        else if(a=="--tracker" && i+1<argc) trackerType = argv[++i];
        else if(a=="--gamelogic" && i+1<argc) gameLogicType = argv[++i];
        else if(a=="--fpscap" && i+1<argc) fpscap = std::stoi(argv[++i]);
#ifdef USE_OLLAMA
        else if(a=="--coaching") enableCoaching = true;
        else if(a=="--coach-personality" && i+1<argc) coachingPersonality = argv[++i];
#endif
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
    std::cout << "  Tracker: " << trackerType << "\n";
    std::cout << "  Game Logic: " << gameLogicType << "\n";
#ifdef USE_OLLAMA
    std::cout << "  AI Coaching: " << (enableCoaching ? "enabled" : "disabled") << "\n";
    if (enableCoaching) {
        std::cout << "  Coach Personality: " << coachingPersonality << "\n";
    }
#endif
    
    Calib calib; calib.load(cfgTable);
    VideoSource vs; if(!vs.open(source)){ std::cerr<<"Failed to open source "<<source<<"\n"; return 1; }
    BallDetector classical;
    classical.loadColors(cfgColors);
    DlDetector dl;
    if(engine=="dl") dl.loadModel("model.onnx");
    
    // Initialize tracking based on selected type
    Tracker legacyTracker; 
    Config tcfg; tcfg.load(cfgTable); 
    legacyTracker.setTableSize({tcfg.getInt("table_width",2540), tcfg.getInt("table_height",1270)});
    
    // Initialize ByteTrack if selected
    std::unique_ptr<pv::modern::ByteTrackMOT> byteTracker;
    std::unique_ptr<pv::ProcessingIsolation> isolation;
    if (trackerType == "bytetrack") {
        pv::modern::ByteTrackMOT::Config config;
        config.trackHighThresh = 0.6f;
        config.trackLowThresh = 0.3f;
        config.maxVelocity = 2000.0f;  // Pool balls can move fast
        config.frameRate = static_cast<int>(vs.fps() > 0 ? vs.fps() : 60);
        byteTracker = std::make_unique<pv::modern::ByteTrackMOT>(config);
        
        isolation = std::make_unique<pv::ProcessingIsolation>();
        isolation->initialize(2, 4);  // 2 GPU cores, 4 CPU cores
        
        std::cout << "  ByteTrack initialized with " << config.frameRate << " fps target\n";
    }
    
    EventEngine events; events.loadTable(cfgTable);
    JsonSink sink;
    
    // Initialize game state and UI
    auto gameState = std::make_shared<GameState>(GameType::EightBall);  // Default to 8-ball
    
    // Initialize modern game logic if selected
    std::unique_ptr<pv::ModernGameLogicAdapter> modernGameLogic;
    if (gameLogicType == "modern") {
        modernGameLogic = std::make_unique<pv::ModernGameLogicAdapter>(gameState.get(), true);
        std::cout << "  Modern game logic initialized with shot segmentation\n";
    }
    
    auto trackerPtr = std::make_shared<Tracker>(legacyTracker);  // Convert to shared_ptr
    UiRenderer uiRenderer;
    OverlayRenderer overlayRenderer(gameState, trackerPtr);
    
#ifdef USE_OLLAMA
    // Initialize coaching system if enabled
    std::unique_ptr<pv::ai::CoachingEngine> coachingEngine;
    if (enableCoaching) {
        auto config = pv::ai::CoachingEngineFactory::getDefaultConfig();
        
        // Set personality based on command line argument
        if (coachingPersonality == "analytical") {
            config.personality = pv::ai::CoachingPrompts::CoachingPersonality::Analytical;
        } else if (coachingPersonality == "challenging") {
            config.personality = pv::ai::CoachingPrompts::CoachingPersonality::Challenging;
        } else if (coachingPersonality == "patient") {
            config.personality = pv::ai::CoachingPrompts::CoachingPersonality::Patient;
        } else if (coachingPersonality == "competitive") {
            config.personality = pv::ai::CoachingPrompts::CoachingPersonality::Competitive;
        } else {
            config.personality = pv::ai::CoachingPrompts::CoachingPersonality::Supportive;  // Default
        }
        
        coachingEngine = std::make_unique<pv::ai::CoachingEngine>(config);
        if (coachingEngine->initialize()) {
            std::cout << "  AI Coaching system initialized successfully\n";
            
            // Set up coaching response callback
            coachingEngine->setResponseCallback([](const pv::ai::CoachingEngine::CoachingResponse& response) {
                if (response.success) {
                    std::cout << "\n[AI Coach]: " << response.advice << "\n";
                } else {
                    std::cout << "\n[AI Coach Error]: " << response.advice << "\n";
                }
            });
            
            // Start coaching session
            coachingEngine->startSession("practice");
        } else {
            std::cout << "  Warning: Failed to initialize AI coaching system\n";
            coachingEngine.reset();
        }
    }
#endif
    
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

        // Run tracking based on selected tracker type
        std::vector<Track> tracks;
        if (trackerType == "bytetrack" && byteTracker) {
            // Convert Ball detections to ByteTrack format
            std::vector<pv::DetectionResult::Detection> byteDetections;
            for (const auto& ball : dets) {
                pv::DetectionResult::Detection det;
                det.x = ball.c.x - ball.r;
                det.y = ball.c.y - ball.r;
                det.w = ball.r * 2;
                det.h = ball.r * 2;
                det.confidence = 0.8f;  // Default confidence for classical detection
                det.classId = 0;        // Pool ball class
                byteDetections.push_back(det);
            }
            
            tracks = byteTracker->update(byteDetections, ts);
        } else {
            // Use legacy tracker
            legacyTracker.update(dets, ts);
            tracks = legacyTracker.tracks();
        }

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
        
        // Process modern game logic if enabled
        if (modernGameLogic && modernGameLogic->isEnabled()) {
            modernGameLogic->processTracks(tracks, ts);
        }
        
#ifdef USE_OLLAMA
        // Process AI coaching if enabled
        if (coachingEngine && coachingEngine->isAvailable()) {
            // Create player info (simplified for demo)
            pv::ai::CoachingPrompts::CoachingContext::PlayerInfo playerInfo;
            playerInfo.skillLevel = "intermediate";  // Could be configurable
            playerInfo.preferredGameType = "8-ball";
            
            // Build game state for coaching
            pv::ai::CoachingEngine::GameState coachingGameState;
            coachingGameState.currentPlayer = (gameState->getCurrentTurn() == PlayerTurn::Player1) ? "Player1" : "Player2";
            coachingGameState.gameType = "8-ball";
            coachingGameState.isGameOver = gameState->isGameOver();
            coachingGameState.ballsRemaining = 15;  // Simplified - could track actual balls
            
            // Check if we should trigger coaching on shots
            for (const auto& event : gameEvents) {
                if (event.type == EventType::Pocket) {
                    // Create a mock shot event for coaching analysis
                    pv::modern::ShotSegmentation::ShotEvent shotEvent;
                    shotEvent.isLegalShot = true;  // Simplified - in real implementation would check rules
                    shotEvent.duration = 5.0f;     // Mock duration
                    shotEvent.shotPower = 0.5f;    // Mock power
                    shotEvent.accuracy = 0.7f;     // Mock accuracy
                    
                    // Trigger shot analysis coaching
                    coachingEngine->processAutoCoaching(shotEvent, coachingGameState, playerInfo);
                }
            }
        }
#endif
        
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
#ifdef USE_OLLAMA
        else if(k=='c' && coachingEngine) {
            // Manual coaching request
            pv::ai::CoachingPrompts::CoachingContext::PlayerInfo playerInfo;
            playerInfo.skillLevel = "intermediate";
            playerInfo.preferredGameType = "8-ball";
            
            coachingEngine->requestDrillRecommendation(playerInfo, {});
        }
#endif
        
        if(fpscap>0) std::this_thread::sleep_for(std::chrono::milliseconds(1000/fpscap));
    }

#ifdef USE_OLLAMA
    // Clean up coaching engine
    if (coachingEngine) {
        coachingEngine->endSession();
        coachingEngine->shutdown();
    }
#endif

    vs.release();
    return 0;
}
