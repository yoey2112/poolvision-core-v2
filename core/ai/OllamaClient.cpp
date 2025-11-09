#include "OllamaClient.hpp"
#include <curl/curl.h>
#include <sstream>
#include <thread>
#include <future>
#include <iostream>

namespace pv {
namespace ai {

OllamaClient::OllamaClient(const OllamaConfig& config) 
    : config_(config), curl_(nullptr), initialized_(false), 
      requestCount_(0), avgResponseTime_(0.0f) {
    initialize();
}

OllamaClient::~OllamaClient() {
    cleanup();
}

OllamaClient::OllamaClient(OllamaClient&& other) noexcept
    : config_(std::move(other.config_)), curl_(other.curl_), 
      initialized_(other.initialized_), requestCount_(other.requestCount_),
      avgResponseTime_(other.avgResponseTime_) {
    other.curl_ = nullptr;
    other.initialized_ = false;
}

OllamaClient& OllamaClient::operator=(OllamaClient&& other) noexcept {
    if (this != &other) {
        cleanup();
        config_ = std::move(other.config_);
        curl_ = other.curl_;
        initialized_ = other.initialized_;
        requestCount_ = other.requestCount_;
        avgResponseTime_ = other.avgResponseTime_;
        
        other.curl_ = nullptr;
        other.initialized_ = false;
    }
    return *this;
}

bool OllamaClient::initialize() {
    if (initialized_) return true;
    
    if (!validateConfig()) {
        return false;
    }
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_ = curl_easy_init();
    
    if (!curl_) {
        return false;
    }
    
    // Set basic options
    curl_easy_setopt(curl_, CURLOPT_USERAGENT, config_.userAgent.c_str());
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, config_.timeout);
    curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT, config_.connectionTimeout);
    curl_easy_setopt(curl_, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl_, CURLOPT_SSL_VERIFYPEER, 0L);  // For local development
    curl_easy_setopt(curl_, CURLOPT_SSL_VERIFYHOST, 0L);
    
    initialized_ = true;
    return true;
}

void OllamaClient::cleanup() {
    if (curl_) {
        curl_easy_cleanup(curl_);
        curl_ = nullptr;
    }
    curl_global_cleanup();
    initialized_ = false;
}

OllamaClient::Response OllamaClient::generateResponse(const std::string& prompt) {
    if (!initialized_ || !curl_) {
        return Response{};
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    nlohmann::json requestData = {
        {"model", config_.model},
        {"prompt", formatPrompt(prompt)},
        {"temperature", config_.temperature},
        {"stream", false}
    };
    
    if (config_.maxTokens > 0) {
        requestData["options"] = nlohmann::json{{"num_predict", config_.maxTokens}};
    }
    
    Response response = performRequest(requestData);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    response.responseTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    
    updateStats(response.responseTime);
    
    return response;
}

void OllamaClient::generateResponseAsync(const std::string& prompt, ResponseCallback callback) {
    std::thread([this, prompt, callback]() {
        auto response = generateResponse(prompt);
        callback(response);
    }).detach();
}

OllamaClient::Response OllamaClient::performRequest(const nlohmann::json& requestData) {
    Response response;
    std::string responseString;
    std::string requestString = requestData.dump();
    
    // Set URL
    std::string url = config_.endpoint + "/api/generate";
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    
    // Set POST data
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, requestString.c_str());
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, requestString.length());
    
    // Set headers
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    
    // Set callbacks
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &responseString);
    
    // Perform request
    CURLcode res = curl_easy_perform(curl_);
    
    // Get response code
    long responseCode;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &responseCode);
    response.responseCode = static_cast<int>(responseCode);
    
    curl_slist_free_all(headers);
    
    if (res == CURLE_OK && responseCode == 200) {
        try {
            nlohmann::json responseJson = parseResponse(responseString);
            if (responseJson.contains("response")) {
                response.content = responseJson["response"].get<std::string>();
                response.success = true;
            } else {
                response.error = "Invalid response format: missing 'response' field";
            }
            
            if (responseJson.contains("done") && responseJson["done"].get<bool>()) {
                response.metadata = responseJson;
            }
        } catch (const nlohmann::json::exception& e) {
            response.error = "JSON parsing error: " + std::string(e.what());
        }
    } else {
        response.error = "HTTP request failed: " + std::string(curl_easy_strerror(res));
        if (responseCode != 200) {
            response.error += " (HTTP " + std::to_string(responseCode) + ")";
        }
    }
    
    return response;
}

bool OllamaClient::isAvailable() const {
    return testConnection();
}

bool OllamaClient::testConnection() const {
    if (!initialized_ || !curl_) {
        return false;
    }
    
    std::string url = config_.endpoint + "/api/tags";
    std::string response;
    
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 5L);  // Quick timeout for availability check
    
    CURLcode res = curl_easy_perform(curl_);
    long responseCode;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &responseCode);
    
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, config_.timeout);  // Reset timeout
    
    return (res == CURLE_OK && responseCode == 200);
}

std::vector<OllamaClient::ModelInfo> OllamaClient::listModels() const {
    std::vector<ModelInfo> models;
    
    if (!initialized_ || !curl_) {
        return models;
    }
    
    std::string url = config_.endpoint + "/api/tags";
    std::string response;
    
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
    
    CURLcode res = curl_easy_perform(curl_);
    long responseCode;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &responseCode);
    
    if (res == CURLE_OK && responseCode == 200) {
        try {
            nlohmann::json responseJson = nlohmann::json::parse(response);
            if (responseJson.contains("models")) {
                for (const auto& modelJson : responseJson["models"]) {
                    ModelInfo info;
                    info.name = modelJson.value("name", "");
                    info.family = modelJson.value("family", "");
                    info.format = modelJson.value("format", "");
                    info.size = modelJson.value("size", 0LL);
                    info.digest = modelJson.value("digest", "");
                    info.modifiedAt = modelJson.value("modified_at", "");
                    models.push_back(info);
                }
            }
        } catch (const nlohmann::json::exception&) {
            // Return empty vector on parse error
        }
    }
    
    return models;
}

bool OllamaClient::modelExists(const std::string& modelName) const {
    auto models = listModels();
    for (const auto& model : models) {
        if (model.name == modelName) {
            return true;
        }
    }
    return false;
}

bool OllamaClient::pullModel(const std::string& modelName) {
    if (!initialized_ || !curl_) {
        return false;
    }
    
    nlohmann::json requestData = {
        {"name", modelName},
        {"stream", false}
    };
    
    std::string url = config_.endpoint + "/api/pull";
    std::string requestString = requestData.dump();
    std::string response;
    
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, requestString.c_str());
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, requestString.length());
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 300L);  // 5 minutes for model download
    
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    
    CURLcode res = curl_easy_perform(curl_);
    long responseCode;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &responseCode);
    
    curl_slist_free_all(headers);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, config_.timeout);  // Reset timeout
    
    return (res == CURLE_OK && responseCode == 200);
}

OllamaClient::Response OllamaClient::getServerVersion() const {
    Response response;
    
    if (!initialized_ || !curl_) {
        return response;
    }
    
    std::string url = config_.endpoint + "/api/version";
    std::string responseString;
    
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &responseString);
    
    CURLcode res = curl_easy_perform(curl_);
    long responseCode;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &responseCode);
    response.responseCode = static_cast<int>(responseCode);
    
    if (res == CURLE_OK && responseCode == 200) {
        response.content = responseString;
        response.success = true;
    } else {
        response.error = "Failed to get server version";
    }
    
    return response;
}

void OllamaClient::setConfig(const OllamaConfig& config) {
    config_ = config;
    if (initialized_ && !validateConfig()) {
        cleanup();
        initialize();
    }
}

void OllamaClient::resetStats() {
    requestCount_ = 0;
    avgResponseTime_ = 0.0f;
}

size_t OllamaClient::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t totalSize = size * nmemb;
    data->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

size_t OllamaClient::HeaderCallback(void* contents, size_t size, size_t nmemb, nlohmann::json* headers) {
    size_t totalSize = size * nmemb;
    std::string header(static_cast<char*>(contents), totalSize);
    
    size_t colonPos = header.find(':');
    if (colonPos != std::string::npos) {
        std::string key = header.substr(0, colonPos);
        std::string value = header.substr(colonPos + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t\r\n"));
        key.erase(key.find_last_not_of(" \t\r\n") + 1);
        value.erase(0, value.find_first_not_of(" \t\r\n"));
        value.erase(value.find_last_not_of(" \t\r\n") + 1);
        
        (*headers)[key] = value;
    }
    
    return totalSize;
}

void OllamaClient::updateStats(float responseTime) const {
    requestCount_++;
    if (requestCount_ == 1) {
        avgResponseTime_ = responseTime;
    } else {
        avgResponseTime_ = (avgResponseTime_ * (requestCount_ - 1) + responseTime) / requestCount_;
    }
    lastRequestTime_ = std::chrono::steady_clock::now();
}

std::string OllamaClient::formatPrompt(const std::string& prompt) const {
    // Basic prompt formatting - can be extended for more sophisticated formatting
    return prompt;
}

nlohmann::json OllamaClient::parseResponse(const std::string& rawResponse) const {
    return nlohmann::json::parse(rawResponse);
}

bool OllamaClient::validateConfig() const {
    if (config_.endpoint.empty()) return false;
    if (config_.model.empty()) return false;
    if (config_.timeout <= 0) return false;
    if (config_.temperature < 0.0f || config_.temperature > 2.0f) return false;
    return true;
}

} // namespace ai
} // namespace pv