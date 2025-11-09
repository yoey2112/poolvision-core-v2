#pragma once
#include "StreamingEngine.hpp"
#include <string>

namespace pv {
namespace streaming {

/**
 * Interface to OBS Studio for overlay injection
 */
class OBSInterface {
public:
    OBSInterface();
    ~OBSInterface();
    
    // Connection management
    bool connect();
    void disconnect();
    bool isConnected() const;
    
    // Streaming control
    bool startStreaming();
    void stopStreaming();
    bool isStreaming() const;
    
    // Overlay management
    bool createOverlaySource(const std::string& sourceName);
    bool updateOverlaySource(const std::string& sourceName, const cv::Mat& overlayImage);
    bool removeOverlaySource(const std::string& sourceName);
    
    // OBS settings
    bool setStreamSettings(const StreamMetadata& metadata);
    bool getStreamStatus();

private:
    bool connected_ = false;
    bool streaming_ = false;
    std::string currentSourceName_;
    
    // OBS WebSocket or plugin communication would go here
    // This is a placeholder for future OBS integration
};

/**
 * Base class for streaming platform APIs
 */
class PlatformAPI {
public:
    virtual ~PlatformAPI() = default;
    
    virtual bool authenticate(const std::string& apiKey) = 0;
    virtual bool isConnected() const = 0;
    virtual bool updateStreamMetadata(const StreamMetadata& metadata) = 0;
    virtual bool startStream() = 0;
    virtual bool stopStream() = 0;

protected:
    bool authenticated_ = false;
    std::string apiKey_;
};

/**
 * Facebook Gaming API integration
 */
class FacebookGamingAPI : public PlatformAPI {
public:
    FacebookGamingAPI();
    ~FacebookGamingAPI();
    
    bool authenticate(const std::string& apiKey) override;
    bool isConnected() const override;
    bool updateStreamMetadata(const StreamMetadata& metadata) override;
    bool startStream() override;
    bool stopStream() override;

private:
    std::string streamKey_;
    std::string accessToken_;
    
    // Facebook Gaming Live API integration would go here
};

/**
 * YouTube Gaming API integration  
 */
class YouTubeGamingAPI : public PlatformAPI {
public:
    YouTubeGamingAPI();
    ~YouTubeGamingAPI();
    
    bool authenticate(const std::string& apiKey) override;
    bool isConnected() const override;
    bool updateStreamMetadata(const StreamMetadata& metadata) override;
    bool startStream() override;
    bool stopStream() override;

private:
    std::string channelId_;
    std::string accessToken_;
    
    // YouTube Live Streaming API integration would go here
};

/**
 * Twitch API integration
 */
class TwitchAPI : public PlatformAPI {
public:
    TwitchAPI();
    ~TwitchAPI();
    
    bool authenticate(const std::string& apiKey) override;
    bool isConnected() const override;
    bool updateStreamMetadata(const StreamMetadata& metadata) override;
    bool startStream() override;
    bool stopStream() override;

private:
    std::string channelId_;
    std::string accessToken_;
    
    // Twitch Helix API integration would go here
};

} // namespace streaming
} // namespace pv