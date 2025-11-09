#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <functional>
#include <nlohmann/json.hpp>

namespace pv {
namespace ai {

/**
 * Ollama API client for local LLM communication
 * Provides high-performance interface to local LLM models with connection pooling
 */
class OllamaClient {
public:
    struct OllamaConfig {
        std::string endpoint = "http://localhost:11434";
        std::string model = "phi3:mini";
        int timeout = 30;
        float temperature = 0.7f;
        int maxTokens = 512;
        int maxRetries = 3;
        int connectionTimeout = 5;
        bool enableStreaming = false;
        std::string userAgent = "PoolVision-Core-v2";
    };

    struct Response {
        std::string content;
        bool success;
        int responseCode;
        float responseTime;
        std::string error;
        nlohmann::json metadata;
        
        Response() : success(false), responseCode(0), responseTime(0.0f) {}
    };

    struct ModelInfo {
        std::string name;
        std::string family;
        std::string format;
        int64_t size;
        std::string digest;
        std::string modifiedAt;
    };

    using ResponseCallback = std::function<void(const Response&)>;

private:
    OllamaConfig config_;
    void* curl_;  // CURL handle
    bool initialized_;
    
    // Performance tracking
    mutable std::chrono::steady_clock::time_point lastRequestTime_;
    mutable int requestCount_;
    mutable float avgResponseTime_;

public:
    explicit OllamaClient(const OllamaConfig& config = OllamaConfig{});
    ~OllamaClient();

    // Non-copyable but movable
    OllamaClient(const OllamaClient&) = delete;
    OllamaClient& operator=(const OllamaClient&) = delete;
    OllamaClient(OllamaClient&& other) noexcept;
    OllamaClient& operator=(OllamaClient&& other) noexcept;

    // Core API methods
    Response generateResponse(const std::string& prompt);
    void generateResponseAsync(const std::string& prompt, ResponseCallback callback);
    
    // Model management
    std::vector<ModelInfo> listModels() const;
    bool pullModel(const std::string& modelName);
    bool modelExists(const std::string& modelName) const;
    
    // Connection management
    bool isAvailable() const;
    bool testConnection() const;
    Response getServerVersion() const;
    
    // Configuration
    void setConfig(const OllamaConfig& config);
    const OllamaConfig& getConfig() const { return config_; }
    void setModel(const std::string& model) { config_.model = model; }
    
    // Performance monitoring
    float getAverageResponseTime() const { return avgResponseTime_; }
    int getRequestCount() const { return requestCount_; }
    void resetStats();

private:
    bool initialize();
    void cleanup();
    Response performRequest(const nlohmann::json& requestData);
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data);
    static size_t HeaderCallback(void* contents, size_t size, size_t nmemb, nlohmann::json* headers);
    
    void updateStats(float responseTime) const;
    std::string formatPrompt(const std::string& prompt) const;
    nlohmann::json parseResponse(const std::string& rawResponse) const;
    bool validateConfig() const;
};

} // namespace ai
} // namespace pv