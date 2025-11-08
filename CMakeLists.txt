cmake_minimum_required(VERSION 3.15)
project(poolvision-core-v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Enable parallel algorithms
add_compile_definitions(TBB_SUPPRESS_DEPRECATED_MESSAGES=1)
if(MSVC)
    add_compile_options(/Zc:__cplusplus /permissive-)
endif()

# Enable testing
enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

option(BUILD_DL_ENGINE "Enable DL detection engine (requires ONNX Runtime)" OFF)

find_package(OpenCV REQUIRED core imgproc highgui videoio imgcodecs calib3d video)
find_package(Eigen3 REQUIRED)

if(BUILD_DL_ENGINE)
  find_package(ONNXRuntime QUIET)
endif()

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

add_library(poolvision_core
  core/util/Types.hpp
  core/util/Config.hpp
  core/util/Config.cpp
  core/util/ColorLab.hpp
  core/util/ColorLab.cpp
  core/util/Timing.hpp
  core/util/Timing.cpp
  core/util/UiRenderer.hpp
  core/util/UiRenderer.cpp
  
  core/game/GameState.hpp
  core/game/GameState.cpp

  core/calib/Calib.hpp
  core/calib/Calib.cpp
  core/calib/Homography.hpp
  core/calib/Homography.cpp

  core/detect/classical/BallDetector.hpp
  core/detect/classical/BallDetector.cpp
  core/detect/classical/StripeClassifier.hpp
  core/detect/classical/StripeClassifier.cpp

  core/detect/dl/DlDetector.hpp
  core/detect/dl/DlDetector.cpp

  core/track/Tracker.hpp
  core/track/Tracker.cpp
  core/track/Physics.hpp
  core/track/Physics.cpp

  core/events/EventEngine.hpp
  core/events/EventEngine.cpp

  core/io/VideoSource.hpp
  core/io/VideoSource.cpp
  core/io/JsonSink.hpp
  core/io/JsonSink.cpp
  
  core/ui/WizardPage.hpp
  core/ui/WizardPage.cpp
  core/ui/WizardManager.hpp
  core/ui/WizardManager.cpp
  core/ui/wizard/CameraSelectionPage.hpp
  core/ui/wizard/CameraSelectionPage.cpp
  core/ui/wizard/CameraOrientationPage.hpp
  core/ui/wizard/CameraOrientationPage.cpp
  core/ui/wizard/TableCalibrationPage.hpp
  core/ui/wizard/TableCalibrationPage.cpp
  core/ui/wizard/TableDimensionsPage.hpp
  core/ui/wizard/TableDimensionsPage.cpp
  core/ui/wizard/CalibrationCompletePage.hpp
  core/ui/wizard/CalibrationCompletePage.cpp
  
  core/ui/UITheme.hpp
  core/ui/UITheme.cpp
  core/ui/menu/MainMenuPage.hpp
  core/ui/menu/MainMenuPage.cpp
  core/ui/menu/SettingsPage.hpp
  core/ui/menu/SettingsPage.cpp
)

target_include_directories(poolvision_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(poolvision_core PUBLIC ${OpenCV_LIBS})

if(BUILD_DL_ENGINE AND ONNXRUNTIME_FOUND)
  target_compile_definitions(poolvision_core PUBLIC USE_DL_ENGINE)
  target_link_libraries(poolvision_core PUBLIC ONNXRUNTIME::ONNXRUNTIME)
endif()

add_executable(table_daemon apps/table_daemon/main.cpp)
target_link_libraries(table_daemon PRIVATE poolvision_core ${OpenCV_LIBS})

add_executable(calibrate apps/calibrate/main.cpp)
target_link_libraries(calibrate PRIVATE poolvision_core ${OpenCV_LIBS})

add_executable(setup_wizard apps/setup_wizard/main.cpp)
target_link_libraries(setup_wizard PRIVATE poolvision_core ${OpenCV_LIBS})

add_executable(pool_vision apps/pool_vision/main.cpp)
target_link_libraries(pool_vision PRIVATE poolvision_core ${OpenCV_LIBS})

# Tests
file(GLOB_RECURSE TEST_SOURCES "tests/unit/*.cpp" "tests/integration/*.cpp")
add_executable(unit_tests ${TEST_SOURCES})
target_link_libraries(unit_tests PRIVATE
    poolvision_core
    ${OpenCV_LIBS}
    GTest::gtest_main
    GTest::gmock_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)

install(TARGETS table_daemon calibrate setup_wizard pool_vision RUNTIME DESTINATION bin)
