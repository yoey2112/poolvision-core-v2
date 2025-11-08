#include <iostream>
#include <memory>
#include "ui/WizardManager.hpp"
#include "ui/wizard/CameraSelectionPage.hpp"
#include "ui/wizard/CameraOrientationPage.hpp"
#include "ui/wizard/TableCalibrationPage.hpp"
#include "ui/wizard/TableDimensionsPage.hpp"
#include "ui/wizard/CalibrationCompletePage.hpp"

using namespace pv;

void printWelcome() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Pool Vision Setup Wizard" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "This wizard will guide you through setting up:" << std::endl;
    std::cout << "  1. Camera selection" << std::endl;
    std::cout << "  2. Camera orientation" << std::endl;
    std::cout << "  3. Table calibration" << std::endl;
    std::cout << "  4. Table dimensions" << std::endl;
    std::cout << std::endl;
    std::cout << "Press ENTER to begin, or Ctrl+C to cancel..." << std::endl;
    std::cin.get();
}

int main(int argc, char** argv) {
    printWelcome();
    
    try {
        // Create wizard manager
        WizardManager wizard;
        
        // Add pages in order
        wizard.addPage(std::make_unique<CameraSelectionPage>());
        wizard.addPage(std::make_unique<CameraOrientationPage>());
        wizard.addPage(std::make_unique<TableCalibrationPage>());
        wizard.addPage(std::make_unique<TableDimensionsPage>());
        wizard.addPage(std::make_unique<CalibrationCompletePage>());
        
        // Run wizard
        std::cout << "\nStarting wizard..." << std::endl;
        bool success = wizard.run();
        
        if (success) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "  Setup completed successfully!" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "\nConfiguration files have been saved." << std::endl;
            std::cout << "You can now run: table_daemon.exe" << std::endl;
            std::cout << std::endl;
            return 0;
        } else {
            std::cout << "\nSetup cancelled by user." << std::endl;
            return 1;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}
