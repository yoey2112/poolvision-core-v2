#include "ShotAnalysisEngine.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <iostream>
#include <sstream>

namespace pv {
namespace ai {
namespace learning {

// NeuralNetwork Implementation
ShotAnalysisEngine::LightweightMLModel::NeuralNetwork::NeuralNetwork(int input, int hidden, int output)
    : inputSize(input), hiddenSize(hidden), outputSize(output) {
    
    // Initialize weights with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> weightDist(0.0f, 0.1f);
    
    // Input to hidden weights
    weights.resize(2);
    weights[0].resize(inputSize * hiddenSize);
    for (auto& w : weights[0]) w = weightDist(gen);
    
    // Hidden to output weights
    weights[1].resize(hiddenSize * outputSize);
    for (auto& w : weights[1]) w = weightDist(gen);
    
    // Biases
    biases.resize(hiddenSize + outputSize);
    for (auto& b : biases) b = weightDist(gen);
}

std::vector<float> ShotAnalysisEngine::LightweightMLModel::NeuralNetwork::forward(const std::vector<float>& input) {
    if (input.size() != static_cast<size_t>(inputSize)) {
        return std::vector<float>(outputSize, 0.0f);
    }
    
    // Hidden layer
    std::vector<float> hidden(hiddenSize, 0.0f);
    for (int h = 0; h < hiddenSize; ++h) {
        for (int i = 0; i < inputSize; ++i) {
            hidden[h] += input[i] * weights[0][i * hiddenSize + h];
        }
        hidden[h] += biases[h];
        hidden[h] = std::tanh(hidden[h]); // Activation function
    }
    
    // Output layer
    std::vector<float> output(outputSize, 0.0f);
    for (int o = 0; o < outputSize; ++o) {
        for (int h = 0; h < hiddenSize; ++h) {
            output[o] += hidden[h] * weights[1][h * outputSize + o];
        }
        output[o] += biases[hiddenSize + o];
        output[o] = 1.0f / (1.0f + std::exp(-output[o])); // Sigmoid activation
    }
    
    return output;
}

void ShotAnalysisEngine::LightweightMLModel::NeuralNetwork::updateWeights(
    const std::vector<float>& input, float target, float learningRate) {
    
    // Simple gradient descent update (simplified for real-time performance)
    auto prediction = forward(input);
    float error = target - prediction[0];
    
    // Update output layer weights (simplified)
    for (int h = 0; h < hiddenSize; ++h) {
        weights[1][h] += learningRate * error * input[std::min(h, inputSize - 1)];
    }
}

// StatisticalModel Implementation
void ShotAnalysisEngine::StatisticalModel::updateStatistics(
    int playerId, const DataCollectionEngine::ShotOutcomeData& shot) {
    
    auto& playerStats = playerStatistics[playerId];
    auto& shotStats = playerStats[shot.shotType];
    
    shotStats.totalAttempts++;
    if (shot.successful) shotStats.successfulShots++;
    
    // Update running averages
    float newSuccessRate = static_cast<float>(shotStats.successfulShots) / shotStats.totalAttempts;
    shotStats.averageSuccessRate = newSuccessRate;
    shotStats.averageDifficulty = (shotStats.averageDifficulty * (shotStats.totalAttempts - 1) + 
                                  shot.shotDifficulty) / shotStats.totalAttempts;
    
    // Update recent performance (last 20 shots)
    shotStats.recentPerformance.push_back(shot.successful ? 1.0f : 0.0f);
    if (shotStats.recentPerformance.size() > 20) {
        shotStats.recentPerformance.erase(shotStats.recentPerformance.begin());
    }
}

float ShotAnalysisEngine::StatisticalModel::predictSuccess(
    int playerId, const DataCollectionEngine::ShotOutcomeData& hypotheticalShot) {
    
    auto playerIt = playerStatistics.find(playerId);
    if (playerIt == playerStatistics.end()) {
        return 0.5f; // No data available, return neutral prediction
    }
    
    auto shotIt = playerIt->second.find(hypotheticalShot.shotType);
    if (shotIt == playerIt->second.end()) {
        return 0.5f; // No data for this shot type
    }
    
    const auto& stats = shotIt->second;
    
    // Base success rate
    float baseRate = stats.averageSuccessRate;
    
    // Adjust for difficulty
    float difficultyAdjustment = 1.0f - (hypotheticalShot.shotDifficulty - stats.averageDifficulty);
    
    // Adjust for recent form
    if (!stats.recentPerformance.empty()) {
        float recentForm = std::accumulate(stats.recentPerformance.begin(), 
                                         stats.recentPerformance.end(), 0.0f) / 
                          stats.recentPerformance.size();
        baseRate = (baseRate + recentForm) / 2.0f; // Weight recent form equally
    }
    
    return std::clamp(baseRate * difficultyAdjustment, 0.0f, 1.0f);
}

// LightweightMLModel Implementation
void ShotAnalysisEngine::LightweightMLModel::trainModel(
    int playerId, const std::vector<DataCollectionEngine::ShotOutcomeData>& trainingData) {
    
    if (trainingData.empty()) return;
    
    // Create or update neural network for this player
    if (playerModels.find(playerId) == playerModels.end()) {
        playerModels[playerId] = std::make_unique<NeuralNetwork>(8, 16, 1); // 8 features, 16 hidden, 1 output
    }
    
    auto& model = playerModels[playerId];
    
    // Train on recent data
    for (const auto& shot : trainingData) {
        std::vector<float> features = {
            shot.shotPosition.x / 1920.0f,
            shot.shotPosition.y / 1080.0f,
            shot.targetPosition.x / 1920.0f,
            shot.targetPosition.y / 1080.0f,
            shot.shotDifficulty,
            shot.shotSpeed / 1000.0f,
            shot.shotAngle / (2.0f * static_cast<float>(M_PI)),
            static_cast<float>(shot.shotType) / 7.0f
        };
        
        float target = shot.successful ? 1.0f : 0.0f;
        model->updateWeights(features, target, 0.01f); // Low learning rate for stability
    }
}

float ShotAnalysisEngine::LightweightMLModel::predictOutcome(
    int playerId, const std::vector<float>& features) {
    
    auto it = playerModels.find(playerId);
    if (it == playerModels.end()) {
        return 0.5f; // No model available
    }
    
    auto prediction = it->second->forward(features);
    return prediction.empty() ? 0.5f : prediction[0];
}

// PatternRecognizer Implementation
void ShotAnalysisEngine::PatternRecognizer::analyzePositionalPatterns(
    int playerId, const std::vector<DataCollectionEngine::ShotOutcomeData>& shots) {
    
    if (shots.empty()) return;
    
    auto& clusters = playerClusters[playerId];
    clusters.clear();
    
    // Simple clustering algorithm for shot positions
    const float clusterRadius = 100.0f; // 100 pixel radius
    
    for (const auto& shot : shots) {
        cv::Point2f position = shot.shotPosition;
        
        // Find nearest cluster
        auto nearestCluster = std::min_element(clusters.begin(), clusters.end(),
            [&position](const PositionalCluster& a, const PositionalCluster& b) {
                float distA = cv::norm(a.center - position);
                float distB = cv::norm(b.center - position);
                return distA < distB;
            });
        
        // Add to nearest cluster if within radius, otherwise create new cluster
        if (nearestCluster != clusters.end() && 
            cv::norm(nearestCluster->center - position) <= clusterRadius) {
            
            // Update cluster
            nearestCluster->center = (nearestCluster->center * nearestCluster->shotCount + position) / 
                                   (nearestCluster->shotCount + 1);
            nearestCluster->shotCount++;
            
            float newSuccessRate = (nearestCluster->successRate * (nearestCluster->shotCount - 1) + 
                                  (shot.successful ? 1.0f : 0.0f)) / nearestCluster->shotCount;
            nearestCluster->successRate = newSuccessRate;
            
            float newDifficulty = (nearestCluster->averageDifficulty * (nearestCluster->shotCount - 1) + 
                                 shot.shotDifficulty) / nearestCluster->shotCount;
            nearestCluster->averageDifficulty = newDifficulty;
            
        } else {
            // Create new cluster
            PositionalCluster newCluster;
            newCluster.center = position;
            newCluster.radius = clusterRadius;
            newCluster.shotCount = 1;
            newCluster.successRate = shot.successful ? 1.0f : 0.0f;
            newCluster.averageDifficulty = shot.shotDifficulty;
            clusters.push_back(newCluster);
        }
    }
}

std::vector<cv::Point2f> ShotAnalysisEngine::PatternRecognizer::findOptimalPositions(
    int playerId, float minSuccessRate) {
    
    std::vector<cv::Point2f> optimalPositions;
    
    auto it = playerClusters.find(playerId);
    if (it == playerClusters.end()) return optimalPositions;
    
    for (const auto& cluster : it->second) {
        if (cluster.successRate >= minSuccessRate && cluster.shotCount >= 3) {
            optimalPositions.push_back(cluster.center);
        }
    }
    
    return optimalPositions;
}

// ShotAnalysisEngine Implementation
ShotAnalysisEngine::ShotAnalysisEngine(DataCollectionEngine* dataCollection)
    : dataCollection_(dataCollection) {
}

ShotAnalysisEngine::~ShotAnalysisEngine() {
    stopAnalysis();
}

void ShotAnalysisEngine::startAnalysis() {
    if (analysisActive_.load()) return;
    
    analysisActive_ = true;
    analysisThread_ = std::thread(&ShotAnalysisEngine::analysisLoop, this);
    
    std::cout << "Shot Analysis Engine started" << std::endl;
}

void ShotAnalysisEngine::stopAnalysis() {
    if (!analysisActive_.load()) return;
    
    analysisActive_ = false;
    updateCondition_.notify_all();
    
    if (analysisThread_.joinable()) {
        analysisThread_.join();
    }
    
    std::cout << "Shot Analysis Engine stopped" << std::endl;
}

ShotAnalysisEngine::ShotAnalysisResult ShotAnalysisEngine::analyzeShotSituation(
    int playerId, const GameState& gameState, const cv::Point2f& cueBallPos, 
    const std::vector<Ball>& targetBalls) {
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    ShotAnalysisResult result;
    
    if (targetBalls.empty()) {
        result.mainPrediction.confidence = 0.0f;
        return result;
    }
    
    // Analyze main target
    Ball mainTarget = targetBalls[0];
    DataCollectionEngine::ShotOutcomeData hypotheticalShot;
    hypotheticalShot.playerId = playerId;
    hypotheticalShot.shotPosition = cueBallPos;
    hypotheticalShot.targetPosition = mainTarget.c;
    hypotheticalShot.shotDifficulty = calculateShotComplexity(hypotheticalShot);
    hypotheticalShot.shotType = DataCollectionEngine::ShotOutcomeData::Straight; // Simplified
    
    // Generate main prediction
    result.mainPrediction = predictShotOutcome(playerId, hypotheticalShot);
    
    // Generate alternative predictions
    for (size_t i = 1; i < std::min(targetBalls.size(), size_t(3)); ++i) {
        hypotheticalShot.targetPosition = targetBalls[i].c;
        hypotheticalShot.shotDifficulty = calculateShotComplexity(hypotheticalShot);
        result.alternatives.push_back(predictShotOutcome(playerId, hypotheticalShot));
    }
    
    // Generate insights
    auto patterns = analyzePlayerPatterns(playerId);
    
    ShotAnalysisResult::LearningInsight insight;
    if (hypotheticalShot.shotDifficulty > 0.7f) {
        insight.category = "opportunity";
        insight.description = "This is a challenging shot that could improve your skills";
        insight.importance = 0.8f;
        insight.recommendations.push_back("Focus on your stance and follow-through");
        result.insights.push_back(insight);
    }
    
    // Set performance context
    result.playerForm = patterns.improvement.overallTrend;
    result.consistencyLevel = 1.0f - patterns.behaviorPattern.confidenceVariability;
    result.recommendedStrategy = generateShotStrategy(playerId, gameState);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double analysisTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    // Update metrics
    metrics_.predictionsGenerated.fetch_add(1);
    double currentAvg = metrics_.avgAnalysisTime.load();
    metrics_.avgAnalysisTime.store(currentAvg * 0.9 + analysisTime * 0.1);
    
    return result;
}

ShotAnalysisEngine::ShotPrediction ShotAnalysisEngine::predictShotOutcome(
    int playerId, const DataCollectionEngine::ShotOutcomeData& hypotheticalShot) {
    
    ShotPrediction prediction;
    
    // Get statistical prediction
    float statPrediction = statisticalModel_.predictSuccess(playerId, hypotheticalShot);
    
    // Get ML prediction if model is available
    std::vector<float> features = extractShotFeatures(hypotheticalShot);
    float mlPrediction = mlModel_.predictOutcome(playerId, features);
    
    // Combine predictions (weight statistical model more heavily for stability)
    float combinedPrediction;
    if (mlModel_.isModelTrained(playerId)) {
        combinedPrediction = 0.7f * statPrediction + 0.3f * mlPrediction;
        prediction.confidence = 0.8f;
    } else {
        combinedPrediction = statPrediction;
        prediction.confidence = 0.6f;
    }
    
    prediction.successProbability = std::clamp(combinedPrediction, 0.0f, 1.0f);
    prediction.difficultyRating = hypotheticalShot.shotDifficulty;
    
    // Generate recommendations
    cv::Point2f direction = hypotheticalShot.targetPosition - hypotheticalShot.shotPosition;
    float distance = cv::norm(direction);
    if (distance > 0) {
        direction /= distance;
        prediction.recommendedContactPoint = hypotheticalShot.targetPosition - direction * 25.0f; // Ball radius
        prediction.recommendedSpeed = direction * std::min(distance * 2.0f, 500.0f); // Reasonable speed
    }
    
    // Generate reasoning
    std::stringstream reasoning;
    reasoning << "Success probability: " << std::fixed << std::setprecision(1) << (prediction.successProbability * 100) << "%";
    if (prediction.difficultyRating > 0.6f) {
        reasoning << " (challenging shot)";
    }
    prediction.reasoning = reasoning.str();
    
    return prediction;
}

ShotAnalysisEngine::PatternAnalysis ShotAnalysisEngine::analyzePlayerPatterns(int playerId) {
    PatternAnalysis analysis;
    
    // Get player data
    auto shotHistory = dataCollection_->getPlayerShotHistory(playerId, 200);
    auto behaviorHistory = dataCollection_->getPlayerBehaviorHistory(playerId);
    
    if (shotHistory.empty()) return analysis;
    
    // Analyze shot type preferences
    std::map<DataCollectionEngine::ShotOutcomeData::ShotType, int> shotTypeCounts;
    std::map<DataCollectionEngine::ShotOutcomeData::ShotType, int> shotTypeSuccesses;
    
    for (const auto& shot : shotHistory) {
        shotTypeCounts[shot.shotType]++;
        if (shot.successful) shotTypeSuccesses[shot.shotType]++;
    }
    
    for (const auto& [shotType, count] : shotTypeCounts) {
        float preference = static_cast<float>(count) / shotHistory.size();
        analysis.shotTypePreferences[shotType] = preference;
        
        if (count > 0) {
            float successRate = static_cast<float>(shotTypeSuccesses[shotType]) / count;
            analysis.shotTypeSuccessRates[shotType] = successRate;
        }
    }
    
    // Analyze positional patterns
    patternRecognizer_.analyzePositionalPatterns(playerId, shotHistory);
    analysis.preferredShotPositions = patternRecognizer_.findOptimalPositions(playerId, 0.7f);
    analysis.problematicPositions = patternRecognizer_.findProblematicPositions(playerId, 0.3f);
    
    // Analyze behavior patterns
    if (!behaviorHistory.empty()) {
        float totalAimingTime = 0;
        float confidenceSum = 0;
        
        for (const auto& behavior : behaviorHistory) {
            totalAimingTime += behavior.aimingTime;
            confidenceSum += behavior.confidenceLevel;
        }
        
        analysis.behaviorPattern.averageAimingTime = totalAimingTime / behaviorHistory.size();
        
        // Calculate confidence variability
        float avgConfidence = confidenceSum / behaviorHistory.size();
        float confidenceVariance = 0;
        for (const auto& behavior : behaviorHistory) {
            confidenceVariance += std::pow(behavior.confidenceLevel - avgConfidence, 2);
        }
        analysis.behaviorPattern.confidenceVariability = std::sqrt(confidenceVariance / behaviorHistory.size());
    }
    
    // Calculate improvement metrics
    if (shotHistory.size() >= 30) {
        int firstHalf = static_cast<int>(shotHistory.size() / 2);
        int firstHalfSuccesses = 0, secondHalfSuccesses = 0;
        
        for (int i = 0; i < firstHalf; ++i) {
            if (shotHistory[i].successful) firstHalfSuccesses++;
        }
        for (size_t i = firstHalf; i < shotHistory.size(); ++i) {
            if (shotHistory[i].successful) secondHalfSuccesses++;
        }
        
        float firstRate = static_cast<float>(firstHalfSuccesses) / firstHalf;
        float secondRate = static_cast<float>(secondHalfSuccesses) / (shotHistory.size() - firstHalf);
        analysis.improvement.overallTrend = secondRate - firstRate;
    }
    
    return analysis;
}

std::vector<ShotAnalysisEngine::ShotRecommendation> ShotAnalysisEngine::generateShotRecommendations(
    int playerId, const GameState& gameState, int maxRecommendations) {
    
    std::vector<ShotRecommendation> recommendations;
    
    // Get player patterns to inform recommendations
    auto patterns = analyzePlayerPatterns(playerId);
    
    // For now, generate simple recommendations based on preferred positions
    for (const auto& position : patterns.preferredShotPositions) {
        if (recommendations.size() >= static_cast<size_t>(maxRecommendations)) break;
        
        ShotRecommendation rec;
        rec.targetBall = position;
        rec.contactPoint = position + cv::Point2f(25.0f, 0); // Simple offset
        rec.recommendedPower = 0.6f; // Moderate power
        rec.expectedSuccess = 0.75f; // High success rate for preferred positions
        rec.strategy = "positional";
        rec.reasoning = "This position has shown good results in your past games";
        
        recommendations.push_back(rec);
    }
    
    return recommendations;
}

void ShotAnalysisEngine::updatePlayerModel(int playerId, 
                                         const DataCollectionEngine::ShotOutcomeData& completedShot) {
    // Update statistical model
    statisticalModel_.updateStatistics(playerId, completedShot);
    
    // Update ML model
    mlModel_.updateModel(playerId, completedShot);
    
    // Schedule background model update
    schedulePlayerUpdate(playerId);
    
    metrics_.modelsUpdated.fetch_add(1);
}

void ShotAnalysisEngine::analysisLoop() {
    while (analysisActive_.load()) {
        std::unique_lock<std::mutex> lock(updateQueueMutex_);
        
        updateCondition_.wait_for(lock, std::chrono::seconds(5), [this] {
            return !analysisActive_.load() || !playersToUpdate_.empty();
        });
        
        if (!analysisActive_.load()) break;
        
        updatePlayerModels();
        
        // Sleep to avoid consuming too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void ShotAnalysisEngine::updatePlayerModels() {
    std::lock_guard<std::mutex> lock(updateQueueMutex_);
    
    while (!playersToUpdate_.empty()) {
        int playerId = playersToUpdate_.front();
        playersToUpdate_.pop();
        
        // Get recent training data
        auto trainingData = dataCollection_->getPlayerShotHistory(playerId, 50);
        if (!trainingData.empty()) {
            mlModel_.trainModel(playerId, trainingData);
        }
        
        // Update patterns
        patternRecognizer_.analyzePositionalPatterns(playerId, trainingData);
    }
}

void ShotAnalysisEngine::schedulePlayerUpdate(int playerId) {
    std::lock_guard<std::mutex> lock(updateQueueMutex_);
    playersToUpdate_.push(playerId);
    updateCondition_.notify_one();
}

std::vector<float> ShotAnalysisEngine::extractShotFeatures(
    const DataCollectionEngine::ShotOutcomeData& shot) {
    
    return {
        shot.shotPosition.x / 1920.0f,
        shot.shotPosition.y / 1080.0f,
        shot.targetPosition.x / 1920.0f,
        shot.targetPosition.y / 1080.0f,
        shot.shotDifficulty,
        shot.shotSpeed / 1000.0f,
        shot.shotAngle / (2.0f * static_cast<float>(M_PI)),
        static_cast<float>(shot.shotType) / 7.0f
    };
}

float ShotAnalysisEngine::calculateShotComplexity(const DataCollectionEngine::ShotOutcomeData& shot) {
    float distance = cv::norm(shot.targetPosition - shot.shotPosition);
    float normalizedDistance = std::min(distance / 500.0f, 1.0f);
    
    float angle = std::abs(shot.shotAngle);
    float normalizedAngle = std::min(angle / (static_cast<float>(M_PI) / 2.0f), 1.0f);
    
    return (normalizedDistance + normalizedAngle) / 2.0f;
}

std::string ShotAnalysisEngine::generateShotStrategy(int playerId, const GameState& gameState) {
    // Simple strategy generation based on game context
    if (gameState.isGameOver()) {
        return "focus";
    }
    
    // Default strategy
    return "balanced";
}

void ShotAnalysisEngine::logAnalysisReport() {
    auto metrics = getAnalysisMetrics();
    
    std::cout << "\n=== Shot Analysis Engine Report ===" << std::endl;
    std::cout << "Predictions generated: " << metrics.predictionsGenerated.load() << std::endl;
    std::cout << "Models updated: " << metrics.modelsUpdated.load() << std::endl;
    std::cout << "Avg prediction time: " << std::fixed << std::setprecision(2) 
              << metrics.avgPredictionTime.load() << "ms" << std::endl;
    std::cout << "Avg analysis time: " << metrics.avgAnalysisTime.load() << "ms" << std::endl;
    std::cout << "===================================" << std::endl;
}

// Factory Implementation
std::unique_ptr<ShotAnalysisEngine> ShotAnalysisFactory::createRealTime(DataCollectionEngine* dataCollection) {
    return std::make_unique<ShotAnalysisEngine>(dataCollection);
}

} // namespace learning
} // namespace ai
} // namespace pv