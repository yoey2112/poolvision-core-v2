cmake_minimum_required(VERSION 3.18)  # Higher version for CUDA support

# Set vcpkg toolchain file
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Performance optimization options
option(ENABLE_GPU_ACCELERATION "Enable GPU acceleration for ball detection and tracking" OFF)
option(ENABLE_CUDA_OPTIMIZATION "Enable CUDA-specific optimizations" ON)
option(BUILD_DL_ENGINE "Enable DL detection engine (requires ONNX Runtime)" OFF)
option(BUILD_STREAMING_SUPPORT "Enable streaming integration with OBS and platform APIs" ON)

# Conditional CUDA language support
if(ENABLE_GPU_ACCELERATION)
    project(poolvision-core-v2 LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    project(poolvision-core-v2 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA configuration for maximum performance
if(ENABLE_GPU_ACCELERATION)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    
    if(NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA Toolkit not found, disabling GPU acceleration")
        set(ENABLE_GPU_ACCELERATION OFF CACHE BOOL "GPU acceleration disabled due to missing CUDA" FORCE)
    endif()
endif()

if(ENABLE_GPU_ACCELERATION AND CUDAToolkit_FOUND)
    
    # CUDA architecture targets (adjust based on target hardware)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")  # Support modern GPUs
    
    # CUDA optimization flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --optimize=3")
    if(ENABLE_CUDA_OPTIMIZATION)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=32")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-O3")
    endif()
    
    # Enable CUDA separable compilation for better optimization
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    
    add_compile_definitions(USE_GPU_ACCELERATION)
    add_compile_definitions(USE_CUDA)
endif()

# Enable OpenMP for CPU parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# CPU performance optimizations
if(MSVC)
    # MSVC optimizations - only apply in Release mode
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /Oi /Ot /GL)  # Maximum optimization
        add_compile_options(/arch:AVX2)       # AVX2 instructions
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")  # Link-time optimization
    endif()
    add_compile_options(/Zc:__cplusplus /permissive- /FS)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # GCC/Clang optimizations - only apply in Release mode
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -mtune=native)
        add_compile_options(-ffast-math -funroll-loops)
        add_compile_options(-flto)  # Link-time optimization
    endif()
endif()

# Enable parallel algorithms
add_compile_definitions(TBB_SUPPRESS_DEPRECATED_MESSAGES=1)

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

# Find packages
# Modern AI and inference options (only available if GPU acceleration is enabled)
option(ENABLE_TENSORRT "Enable TensorRT inference engine" OFF)
option(ENABLE_OLLAMA_INTEGRATION "Enable Ollama LLM coaching" ON)
option(ENABLE_NVDEC "Enable NVDEC hardware video decoding" OFF)

# Find modern packages
if(ENABLE_GPU_ACCELERATION AND CUDAToolkit_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS 
        core imgproc highgui videoio imgcodecs calib3d video
        # GPU-accelerated components
        cudaimgproc cudaarithm cudafeatures2d cudawarping
    )
else()
    find_package(OpenCV REQUIRED COMPONENTS 
        core imgproc highgui videoio imgcodecs calib3d video
    )
endif()
find_package(Eigen3 REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)

# Modern dependencies - only check if GPU acceleration is enabled
if(ENABLE_GPU_ACCELERATION AND CUDAToolkit_FOUND)
    if(ENABLE_TENSORRT)
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include)
        find_library(TENSORRT_LIBRARY nvinfer HINTS ${TENSORRT_ROOT} PATH_SUFFIXES lib)
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
            add_compile_definitions(USE_TENSORRT)
            message(STATUS "TensorRT found and enabled")
        else()
            message(STATUS "TensorRT not found, GPU inference will use fallback")
            set(ENABLE_TENSORRT OFF)
        endif()
    endif()
    
    if(ENABLE_NVDEC)
        find_library(NVDECODE_LIBRARY nvidia-encode HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib)
        if(NVDECODE_LIBRARY)
            add_compile_definitions(USE_NVDEC)
            message(STATUS "NVDEC found and enabled")
        else()
            message(STATUS "NVDEC not found, using standard video capture")
        endif()
    endif()
endif()

# Ollama LLM integration (works independently of GPU acceleration)
if(ENABLE_OLLAMA_INTEGRATION)
    find_package(CURL QUIET)
    find_package(nlohmann_json CONFIG QUIET)
    if(CURL_FOUND AND nlohmann_json_FOUND)
        add_compile_definitions(USE_OLLAMA)
        message(STATUS "Ollama integration enabled")
    else()
        message(STATUS "CURL or nlohmann_json not found, disabling Ollama integration")
        set(ENABLE_OLLAMA_INTEGRATION OFF)
    endif()
endif()

if(BUILD_DL_ENGINE)
  find_package(ONNXRuntime QUIET)
endif()

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

# GPU-accelerated core library
add_library(poolvision_core
  # Core utilities (CPU)
  core/util/Types.hpp
  core/util/Config.hpp
  core/util/Config.cpp
  core/util/ColorLab.hpp
  core/util/ColorLab.cpp
  core/util/Timing.hpp
  core/util/Timing.cpp
  core/util/UiRenderer.hpp
  core/util/UiRenderer.cpp
  core/util/UserConfig.hpp
  core/util/UserConfig.cpp
  
  # Performance monitoring and processing isolation (needed for both CPU and GPU)
  core/performance/ProcessingIsolation.hpp
  core/performance/ProcessingIsolation.cpp
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/util/GpuPerformanceMonitor.hpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/util/GpuPerformanceMonitor.cpp>
  
  # Game state management (CPU)
  core/game/GameState.hpp
  core/game/GameState.cpp
  core/game/ModernGameLogicAdapter.hpp

  # Calibration (CPU)
  core/calib/Calib.hpp
  core/calib/Calib.cpp
  core/calib/Homography.hpp
  core/calib/Homography.cpp

  # Ball detection - Classical CPU implementation
  core/detect/classical/BallDetector.hpp
  core/detect/classical/BallDetector.cpp
  core/detect/classical/StripeClassifier.hpp
  core/detect/classical/StripeClassifier.cpp

  # Modern GPU detection pipeline
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/detect/modern/TensorRtBallDetector.hpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/detect/modern/TensorRtBallDetector.cpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/detect/modern/CudaPreprocessKernels.hpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/detect/modern/CudaPreprocessKernels.cpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/detect/modern/CudaPreprocessKernels.cu>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/detect/modern/GpuNonMaxSuppression.hpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/detect/modern/GpuNonMaxSuppression.cpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/detect/modern/GpuNonMaxSuppression.cu>
  
  # Modern CPU tracking pipeline  
  core/track/modern/ByteTrackMOT.hpp
  core/track/modern/ByteTrackMOT.cpp
  
  # Modern game rules engine
  core/game/modern/ShotSegmentation.hpp
  core/game/modern/ShotSegmentation.cpp
  
  # Modern LLM coaching system
  $<$<BOOL:${ENABLE_OLLAMA_INTEGRATION}>:core/ai/OllamaClient.hpp>
  $<$<BOOL:${ENABLE_OLLAMA_INTEGRATION}>:core/ai/OllamaClient.cpp>
  $<$<BOOL:${ENABLE_OLLAMA_INTEGRATION}>:core/ai/CoachingPrompts.hpp>
  $<$<BOOL:${ENABLE_OLLAMA_INTEGRATION}>:core/ai/CoachingPrompts.cpp>
  $<$<BOOL:${ENABLE_OLLAMA_INTEGRATION}>:core/ai/CoachingEngine.hpp>
  $<$<BOOL:${ENABLE_OLLAMA_INTEGRATION}>:core/ai/CoachingEngine.cpp>
  
  # Modern UI rendering pipeline
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/ui/modern/SeparatedUIRenderer.hpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/ui/modern/SeparatedUIRenderer.cpp>
  
  # Modern pipeline integration
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/integration/ModernPipelineIntegrator.hpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/integration/ModernPipelineIntegrator.cpp>
  
  # Legacy GPU-accelerated components (fallback)
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/detect/gpu/GpuBallDetector.hpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/detect/gpu/GpuBallDetector.cpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/detect/gpu/GpuColorClassifier.hpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/detect/gpu/GpuColorClassifier.cpp>

  # Deep learning detection (optional)
  core/detect/dl/DlDetector.hpp
  core/detect/dl/DlDetector.cpp

  # Ball tracking - Classical CPU implementation
  core/track/Tracker.hpp
  core/track/Tracker.cpp
  core/track/Physics.hpp
  core/track/Physics.cpp

  # Ball tracking - GPU accelerated implementation
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/track/gpu/GpuTracker.hpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/track/gpu/GpuTracker.cpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/track/gpu/GpuKalmanFilter.hpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/track/gpu/GpuKalmanFilter.cpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/track/gpu/GpuHungarianSolver.hpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/track/gpu/GpuHungarianSolver.cpp>

  # Events processing (CPU)
  core/events/EventEngine.hpp
  core/events/EventEngine.cpp

  # High-performance I/O
  core/io/VideoSource.hpp
  core/io/VideoSource.cpp
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/io/gpu/HighPerformanceVideoSource.hpp>
  $<$<BOOL:${ENABLE_GPU_ACCELERATION}>:core/io/gpu/HighPerformanceVideoSource.cpp>
  core/io/JsonSink.hpp
  core/io/JsonSink.cpp
  core/video/SessionVideoManager.hpp
  core/video/SessionVideoManager.cpp
  
  # AI Learning System (Simplified for reliable compilation)
  core/ai/learning/SimpleDataCollectionEngine.hpp
  core/ai/learning/SimpleDataCollectionEngine.cpp
  core/ai/learning/SimpleAILearningSystem.hpp
  core/ai/learning/SimpleAILearningSystem.cpp
  
  # Streaming Integration (Phase 10.2)
  core/streaming/StreamingEngine.hpp
  core/streaming/StreamingEngine.cpp
  core/streaming/OverlayManager.hpp
  core/streaming/OverlayManager.cpp
  core/streaming/TemplateSystem.hpp
  core/streaming/TemplateSystem.cpp
  core/streaming/OBSInterface.hpp
  core/streaming/OBSInterface.cpp
  core/streaming/PlatformAPIs.hpp
  core/streaming/PlatformAPIs.cpp
  
  # UI components (CPU)
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
  core/ui/ResponsiveLayout.hpp
  core/ui/ResponsiveLayout.cpp
  core/ui/menu/MainMenuPage.hpp
  core/ui/menu/MainMenuPage.cpp
  core/ui/menu/SettingsPage.hpp
  core/ui/menu/SettingsPage.cpp
  core/ui/menu/PlayerProfilesPage.hpp
  core/ui/menu/PlayerProfilesPage.cpp
  
  # Database (CPU)
  core/db/PlayerProfile.hpp
  core/db/Database.hpp
  core/db/Database.cpp
  
  # Overlay rendering (CPU with GPU data)
  core/ui/OverlayRenderer.hpp
  core/ui/OverlayRenderer.cpp
  
  # Game components (CPU)
  core/game/GameRecorder.hpp
  core/game/GameRecorder.cpp
  core/game/SessionPlayback.hpp
  core/game/SessionPlayback.cpp
  core/game/TrainingMode.hpp
  core/game/TrainingMode.cpp
  core/game/ShotLibrary.hpp
  core/game/ShotLibrary.cpp
  core/game/DrillSystem.hpp
  core/game/DrillSystem.cpp
  core/game/DrillLibrary.hpp
  core/game/DrillLibrary.cpp
  core/game/MatchSystem.hpp
  core/game/MatchSystem.cpp
  
  # Analytics UI (CPU)
  core/ui/menu/AnalyticsPage.hpp
  core/ui/menu/AnalyticsPage.cpp
  core/ui/menu/DrillsPage.hpp
  core/ui/menu/DrillsPage.cpp
  
  # Match UI (CPU)
  core/ui/MatchUI.hpp
  core/ui/MatchUI.cpp
)

target_include_directories(poolvision_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(poolvision_core PUBLIC ${OpenCV_LIBS} unofficial::sqlite3::sqlite3)

# Modern technology integration
if(ENABLE_GPU_ACCELERATION)
    target_link_libraries(poolvision_core PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
    target_compile_definitions(poolvision_core PUBLIC USE_GPU_ACCELERATION)
    
    # TensorRT integration
    if(ENABLE_TENSORRT)
        target_include_directories(poolvision_core PRIVATE ${TENSORRT_INCLUDE_DIR})
        target_link_libraries(poolvision_core PUBLIC ${TENSORRT_LIBRARY})
        target_compile_definitions(poolvision_core PUBLIC USE_TENSORRT)
    endif()
    
    # NVDEC integration
    if(ENABLE_NVDEC AND NVDECODE_LIBRARY)
        target_link_libraries(poolvision_core PUBLIC ${NVDECODE_LIBRARY})
        target_compile_definitions(poolvision_core PUBLIC USE_NVDEC)
    endif()
    
    # Set CUDA properties
    set_property(TARGET poolvision_core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET poolvision_core PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# Ollama LLM integration
if(ENABLE_OLLAMA_INTEGRATION)
    target_link_libraries(poolvision_core PUBLIC CURL::libcurl nlohmann_json::nlohmann_json)
    target_compile_definitions(poolvision_core PUBLIC USE_OLLAMA)
endif()

# Legacy GPU acceleration support
if(ENABLE_GPU_ACCELERATION)
    target_link_libraries(poolvision_core PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
    target_compile_definitions(poolvision_core PUBLIC USE_GPU_ACCELERATION)
    
    # Set CUDA properties
    set_property(TARGET poolvision_core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET poolvision_core PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# DL engine support
if(BUILD_DL_ENGINE AND ONNXRUNTIME_FOUND)
  target_compile_definitions(poolvision_core PUBLIC USE_DL_ENGINE)
  target_link_libraries(poolvision_core PUBLIC ONNXRUNTIME::ONNXRUNTIME)
  
  # Enable CUDA support for ONNX Runtime if available
  if(ENABLE_GPU_ACCELERATION)
    target_compile_definitions(poolvision_core PUBLIC USE_ONNX_CUDA)
  endif()
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

# Streaming support
if(BUILD_STREAMING_SUPPORT)
    # OBS plugin (optional - requires OBS development libraries)
    add_subdirectory(plugins/obs OPTIONAL)
    
    message(STATUS "Streaming support enabled")
else()
    message(STATUS "Streaming support disabled")
endif()
