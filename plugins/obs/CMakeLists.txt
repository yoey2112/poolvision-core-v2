cmake_minimum_required(VERSION 3.16)

project(poolvision_obs_plugin VERSION 2.1.1)

# Find OBS Studio
find_package(LibObs QUIET)
if(NOT LibObs_FOUND)
    message(WARNING "OBS Studio not found - OBS plugin will not be built")
    return()
endif()

# Find Qt5 for OBS plugin UI
find_package(Qt5 COMPONENTS Core Widgets QUIET)
if(NOT Qt5_FOUND)
    message(WARNING "Qt5 not found - OBS plugin will not be built")
    return()
endif()

# Plugin source files
set(PLUGIN_SOURCES
    PoolVisionOBS.cpp
    PoolVisionOBSPlugin.cpp
    OverlaySource.cpp
    PluginSettings.cpp
)

set(PLUGIN_HEADERS
    PoolVisionOBS.hpp
    OverlaySource.hpp
    PluginSettings.hpp
)

# Create the plugin
add_library(poolvision_obs_plugin MODULE
    ${PLUGIN_SOURCES}
    ${PLUGIN_HEADERS}
)

# Link with OBS and Qt
target_link_libraries(poolvision_obs_plugin
    libobs
    Qt5::Core
    Qt5::Widgets
    poolvision_streaming
)

# Include directories
target_include_directories(poolvision_obs_plugin PRIVATE
    ${LIBOBS_INCLUDE_DIR}
    ../../core/streaming
)

# Plugin properties
set_target_properties(poolvision_obs_plugin PROPERTIES
    FOLDER "plugins"
    PREFIX ""
    SUFFIX ".dll"
)

# Windows specific settings
if(WIN32)
    target_compile_definitions(poolvision_obs_plugin PRIVATE
        _WIN32_WINNT=0x0601
        WIN32_LEAN_AND_MEAN
    )
    
    # Export definitions
    set_target_properties(poolvision_obs_plugin PROPERTIES
        LINK_FLAGS "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/obs_plugin.def"
    )
endif()

# Installation
install(TARGETS poolvision_obs_plugin
    DESTINATION obs-plugins/64bit
)

# Install data files
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/data/locale/en-US.ini
    DESTINATION obs-plugins/64bit/locale
)

message(STATUS "Pool Vision OBS Plugin configured successfully")