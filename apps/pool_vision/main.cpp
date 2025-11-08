#include "../../core/ui/menu/MainMenuPage.hpp"
#include "../../core/ui/menu/SettingsPage.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace pv;

/**
 * @brief Main application class for Pool Vision
 * 
 * Manages the main menu and navigation between different modules
 */
class PoolVisionApp {
public:
    PoolVisionApp()
        : windowName_("Pool Vision")
        , windowWidth_(1280)
        , windowHeight_(720)
        , running_(false) {
    }
    
    void run() {
        // Create window
        cv::namedWindow(windowName_, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName_, windowWidth_, windowHeight_);
        
        // Initialize main menu
        mainMenu_.init();
        mainMenu_.setWindowSize(windowWidth_, windowHeight_);
        
        settingsPage_.init();
        settingsPage_.setWindowSize(windowWidth_, windowHeight_);
        
        running_ = true;
        
        // Set mouse callback
        cv::setMouseCallback(windowName_, PoolVisionApp::onMouse, this);
        
        // Main loop
        while (running_) {
            cv::Mat frame;
            
            switch (currentState_) {
                case State::MainMenu:
                    frame = mainMenu_.render();
                    handleMainMenuAction();
                    break;
                    
                case State::Settings:
                    frame = settingsPage_.render();
                    if (settingsPage_.shouldGoBack()) {
                        settingsPage_.resetGoBack();
                        currentState_ = State::MainMenu;
                    }
                    break;
                    
                case State::NewGame:
                    showPlaceholder(frame, "New Game");
                    break;
                    
                case State::Drills:
                    showPlaceholder(frame, "Drills & Practice");
                    break;
                    
                case State::PlayerProfiles:
                    showPlaceholder(frame, "Player Profiles");
                    break;
                    
                case State::Analytics:
                    showPlaceholder(frame, "Analytics Dashboard");
                    break;
                    
                case State::Calibration:
                    showPlaceholder(frame, "Calibration");
                    break;
            }
            
            if (!frame.empty()) {
                cv::imshow(windowName_, frame);
            }
            
            // Handle keyboard
            int key = cv::waitKey(30);
            if (key != -1) {
                handleKeyboard(key);
            }
        }
        
        cv::destroyWindow(windowName_);
    }
    
private:
    enum class State {
        MainMenu,
        NewGame,
        Drills,
        PlayerProfiles,
        Analytics,
        Settings,
        Calibration
    };
    
    void handleMainMenuAction() {
        MenuAction action = mainMenu_.getSelectedAction();
        
        if (action == MenuAction::None) {
            return;
        }
        
        mainMenu_.resetAction();
        
        switch (action) {
            case MenuAction::NewGame:
                std::cout << "Starting New Game..." << std::endl;
                currentState_ = State::NewGame;
                break;
                
            case MenuAction::Drills:
                std::cout << "Opening Drills & Practice..." << std::endl;
                currentState_ = State::Drills;
                break;
                
            case MenuAction::PlayerProfiles:
                std::cout << "Opening Player Profiles..." << std::endl;
                currentState_ = State::PlayerProfiles;
                break;
                
            case MenuAction::Analytics:
                std::cout << "Opening Analytics Dashboard..." << std::endl;
                currentState_ = State::Analytics;
                break;
                
            case MenuAction::Settings:
                std::cout << "Opening Settings..." << std::endl;
                currentState_ = State::Settings;
                break;
                
            case MenuAction::Calibration:
                std::cout << "Opening Calibration..." << std::endl;
                // Launch setup wizard
                launchSetupWizard();
                break;
                
            case MenuAction::Exit:
                std::cout << "Exiting Pool Vision..." << std::endl;
                running_ = false;
                break;
                
            default:
                break;
        }
    }
    
    void handleKeyboard(int key) {
        // Forward keyboard events to current page
        switch (currentState_) {
            case State::MainMenu:
                mainMenu_.onKey(key);
                break;
                
            case State::Settings:
                settingsPage_.onKey(key);
                break;
                
            default:
                // ESC to return to main menu from any screen
                if (key == 27) {
                    currentState_ = State::MainMenu;
                }
                break;
        }
    }
    
    void showPlaceholder(cv::Mat& frame, const std::string& title) {
        frame = cv::Mat(windowHeight_, windowWidth_, CV_8UC3, UITheme::Colors::DarkBg);
        
        // Draw title
        UITheme::drawTitleBar(frame, title, 80);
        
        // Draw "Coming Soon" message
        std::string message = "This feature is coming in Phase 3+";
        cv::Size textSize = UITheme::getTextSize(message, UITheme::Fonts::FontFace,
                                                 UITheme::Fonts::HeadingSize,
                                                 UITheme::Fonts::HeadingThickness);
        cv::Point textPos((windowWidth_ - textSize.width) / 2,
                         windowHeight_ / 2);
        
        UITheme::drawTextWithShadow(frame, message, textPos, UITheme::Fonts::FontFace,
                                   UITheme::Fonts::HeadingSize,
                                   UITheme::Colors::TextSecondary,
                                   UITheme::Fonts::HeadingThickness);
        
        // Draw "Press ESC to return" message
        std::string hint = "Press ESC to return to main menu";
        cv::Size hintSize = UITheme::getTextSize(hint, UITheme::Fonts::FontFace,
                                                 UITheme::Fonts::BodySize,
                                                 UITheme::Fonts::BodyThickness);
        cv::Point hintPos((windowWidth_ - hintSize.width) / 2,
                         windowHeight_ / 2 + 50);
        
        cv::putText(frame, hint, hintPos, UITheme::Fonts::FontFace,
                   UITheme::Fonts::BodySize, UITheme::Colors::TextDisabled,
                   UITheme::Fonts::BodyThickness);
    }
    
    void launchSetupWizard() {
        std::cout << "Launching Setup Wizard..." << std::endl;
        
        // On Windows, launch the setup wizard executable
#ifdef _WIN32
        system("start /B build\\Debug\\setup_wizard.exe");
#else
        system("./build/Debug/setup_wizard &");
#endif
        
        // Return to main menu
        currentState_ = State::MainMenu;
    }
    
    static void onMouse(int event, int x, int y, int flags, void* userdata) {
        PoolVisionApp* app = static_cast<PoolVisionApp*>(userdata);
        
        // Forward mouse events to current page
        switch (app->currentState_) {
            case State::MainMenu:
                app->mainMenu_.onMouse(event, x, y, flags);
                break;
                
            case State::Settings:
                app->settingsPage_.onMouse(event, x, y, flags);
                break;
                
            default:
                break;
        }
    }
    
    std::string windowName_;
    int windowWidth_;
    int windowHeight_;
    bool running_;
    State currentState_ = State::MainMenu;
    
    MainMenuPage mainMenu_;
    SettingsPage settingsPage_;
};

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Pool Vision Core v2" << std::endl;
    std::cout << "  Phase 2: Main Menu & Settings" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    try {
        PoolVisionApp app;
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
