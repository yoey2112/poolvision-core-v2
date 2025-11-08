#include <gtest/gtest.h>
#include "../../core/util/Config.hpp"
#include <sstream>

using namespace pv;

TEST(ConfigTest, LoadValidConfig) {
    Config cfg;
    std::stringstream ss;
    ss << "width: 1280\n";
    ss << "height: 720\n";
    ss << "fps: 30\n";
    ss << "array: [1.0, 2.0, 3.0]\n";
    
    EXPECT_TRUE(cfg.load("test_config.yaml"));
    EXPECT_EQ(cfg.getInt("width"), 1280);
    EXPECT_EQ(cfg.getInt("height"), 720);
    EXPECT_EQ(cfg.getInt("fps"), 30);
    
    auto arr = cfg.getArray("array");
    ASSERT_EQ(arr.size(), 3);
    EXPECT_FLOAT_EQ(arr[0], 1.0);
    EXPECT_FLOAT_EQ(arr[1], 2.0);
    EXPECT_FLOAT_EQ(arr[2], 3.0);
}

TEST(ConfigTest, ValidationRules) {
    Config cfg;
    // Load test config
    std::stringstream ss;
    ss << "width: 1280\n";
    ss << "height: 720\n";
    ss << "fps: 30\n";
    
    EXPECT_TRUE(cfg.validateFile("camera"));
    
    // Test invalid config
    Config invalid;
    std::stringstream ss2;
    ss2 << "width: 100\n"; // too small
    ss2 << "height: 10000\n"; // too large
    ss2 << "fps: 200\n"; // too high
    
    EXPECT_FALSE(invalid.validateFile("camera"));
}

TEST(ConfigTest, DefaultValues) {
    Config cfg;
    EXPECT_EQ(cfg.getInt("nonexistent", 42), 42);
    EXPECT_EQ(cfg.getString("nonexistent", "default"), "default");
    EXPECT_FLOAT_EQ(cfg.getDouble("nonexistent", 3.14), 3.14);
}