#pragma once

#include <memory>
#include <vector>
#include <map>
#include <queue>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>

#include "DataCollectionEngine.hpp"
#include "../../track/modern/ByteTrackMOT.hpp"
#include "../../game/modern/ShotSegmentation.hpp"
#include "../../util/Types.hpp"

namespace pv {
namespace ai {
namespace learning {

/**
 * Shot Analysis and Learning System
 * 
 * Analyzes shot patterns, predicts outcomes, and provides
 * real-time shot recommendations using machine learning models.
 * CPU-optimized to work alongside the GPU inference pipeline.
 */
class ShotAnalysisEngine {
public:
    struct ShotPrediction {
        float successProbability;              // 0-1 probability of success
        float difficultyRating;                // 0-1 difficulty assessment
        cv::Point2f recommendedContactPoint;   // Optimal contact point
        cv::Point2f recommendedSpeed;          // Recommended velocity vector
        float confidence;                      // Model confidence 0-1
        std::string reasoning;                 // Human-readable explanation
        
        // Alternative suggestions
        std::vector<cv::Point2f> alternativeTargets;
        std::vector<float> alternativeProbabilities;
        
        ShotPrediction() : successProbability(0), difficultyRating(0),
                         confidence(0) {}
    };
    
    struct ShotAnalysisResult {
        ShotPrediction mainPrediction;
        std::vector<ShotPrediction> alternatives;
        
        // Learning insights
        struct LearningInsight {
            std::string category;              // "weakness", "strength", "opportunity"
            std::string description;           // Human-readable insight
            float importance;                  // 0-1 importance score
            std::vector<std::string> recommendations;
        };
        
        std::vector<LearningInsight> insights;
        
        // Performance context
        float playerForm;                      // Current form 0-1
        float consistencyLevel;                // Consistency rating 0-1
        std::string recommendedStrategy;       // Strategic recommendation
    };
    
    struct PatternAnalysis {
        // Shot type patterns
        std::map<DataCollectionEngine::ShotOutcomeData::ShotType, float> shotTypePreferences;
        std::map<DataCollectionEngine::ShotOutcomeData::ShotType, float> shotTypeSuccessRates;
        
        // Position patterns
        std::vector<cv::Point2f> preferredShotPositions;
        std::vector<cv::Point2f> problematicPositions;
        
        // Behavioral patterns
        struct BehaviorPattern {
            float averageAimingTime;
            float confidenceVariability;
            std::vector<float> performanceByTimeOfDay;
            std::vector<float> performanceBySessionLength;
            float pressurePerformance;         // Performance under pressure
        };
        
        BehaviorPattern behaviorPattern;
        
        // Improvement tracking
        struct ImprovementMetrics {
            float overallTrend;                // Positive = improving
            std::map<DataCollectionEngine::ShotOutcomeData::ShotType, float> shotTypeTrends;
            float learningRate;                // How quickly player learns
            std::vector<std::string> recentMilestones;
        };
        
        ImprovementMetrics improvement;
    };

private:
    // Statistical learning models
    class StatisticalModel {
    public:
        struct ShotStatistics {
            int totalAttempts;
            int successfulShots;
            float averageDifficulty;
            float averageSuccessRate;
            std::vector<float> recentPerformance;
            
            ShotStatistics() : totalAttempts(0), successfulShots(0),
                             averageDifficulty(0), averageSuccessRate(0) {}
        };
        
        std::map<int, std::map<DataCollectionEngine::ShotOutcomeData::ShotType, ShotStatistics>> playerStatistics;
        
    public:
        void updateStatistics(int playerId, const DataCollectionEngine::ShotOutcomeData& shot);
        float predictSuccess(int playerId, const DataCollectionEngine::ShotOutcomeData& hypotheticalShot);
        ShotStatistics getPlayerShotStats(int playerId, DataCollectionEngine::ShotOutcomeData::ShotType shotType);
        std::vector<DataCollectionEngine::ShotOutcomeData::ShotType> getPreferredShotTypes(int playerId);
    };
    
    // Machine learning models (lightweight for real-time use)
    class LightweightMLModel {
    private:
        struct NeuralNetwork {
            std::vector<std::vector<float>> weights;
            std::vector<float> biases;
            int inputSize;
            int hiddenSize;
            int outputSize;
            
            NeuralNetwork(int input, int hidden, int output);
            std::vector<float> forward(const std::vector<float>& input);
            void updateWeights(const std::vector<float>& input, float target, float learningRate);
        };
        
        std::map<int, std::unique_ptr<NeuralNetwork>> playerModels;
        
    public:
        void trainModel(int playerId, const std::vector<DataCollectionEngine::ShotOutcomeData>& trainingData);
        float predictOutcome(int playerId, const std::vector<float>& features);
        void updateModel(int playerId, const DataCollectionEngine::ShotOutcomeData& newData);
        bool isModelTrained(int playerId);
    };
    
    // Pattern recognition system
    class PatternRecognizer {
    private:
        struct PositionalCluster {
            cv::Point2f center;
            float radius;
            int shotCount;
            float successRate;
            float averageDifficulty;
        };
        
        std::map<int, std::vector<PositionalCluster>> playerClusters;
        
    public:
        void analyzePositionalPatterns(int playerId, const std::vector<DataCollectionEngine::ShotOutcomeData>& shots);
        std::vector<cv::Point2f> findOptimalPositions(int playerId, float minSuccessRate = 0.7f);
        std::vector<cv::Point2f> findProblematicPositions(int playerId, float maxSuccessRate = 0.3f);
        float getPositionalDifficulty(int playerId, const cv::Point2f& position);
    };
    
    // Real-time analysis components
    StatisticalModel statisticalModel_;
    LightweightMLModel mlModel_;
    PatternRecognizer patternRecognizer_;
    
    // Data sources
    DataCollectionEngine* dataCollection_;
    
    // Performance optimization
    std::atomic<bool> analysisActive_{false};
    std::thread analysisThread_;
    std::queue<int> playersToUpdate_;
    std::mutex updateQueueMutex_;
    std::condition_variable updateCondition_;

public:
    explicit ShotAnalysisEngine(DataCollectionEngine* dataCollection);
    ~ShotAnalysisEngine();
    
    // Lifecycle management
    void startAnalysis();
    void stopAnalysis();
    bool isAnalysisActive() const { return analysisActive_.load(); }
    
    // Real-time shot analysis
    ShotAnalysisResult analyzeShotSituation(int playerId, const GameState& gameState,
                                          const cv::Point2f& cueBallPos,
                                          const std::vector<Ball>& targetBalls);
    
    ShotPrediction predictShotOutcome(int playerId, const DataCollectionEngine::ShotOutcomeData& hypotheticalShot);
    
    // Pattern analysis
    PatternAnalysis analyzePlayerPatterns(int playerId);
    std::vector<cv::Point2f> recommendOptimalPositions(int playerId, const GameState& gameState);
    
    // Learning and adaptation
    void updatePlayerModel(int playerId, const DataCollectionEngine::ShotOutcomeData& completedShot);
    void triggerModelRetraining(int playerId);
    
    // Shot recommendations
    struct ShotRecommendation {
        cv::Point2f targetBall;
        cv::Point2f contactPoint;
        float recommendedPower;         // 0-1 power level
        float expectedSuccess;          // 0-1 success probability
        std::string strategy;           // "aggressive", "safe", "positional"
        std::string reasoning;          // Why this shot is recommended
    };
    
    std::vector<ShotRecommendation> generateShotRecommendations(int playerId, 
                                                               const GameState& gameState,
                                                               int maxRecommendations = 3);
    
    // Performance analysis
    struct PerformanceAssessment {
        float currentForm;              // 0-1 current performance level
        float consistency;              // 0-1 consistency rating
        std::vector<std::string> strengths;
        std::vector<std::string> weaknesses;
        std::vector<std::string> recommendations;
        float improvementRate;          // Rate of improvement
        float skillLevel;               // Overall skill assessment 0-1
    };
    
    PerformanceAssessment assessPlayerPerformance(int playerId);
    
    // Insights and coaching integration
    std::vector<std::string> generateLearningInsights(int playerId);
    std::string explainShotDifficulty(const DataCollectionEngine::ShotOutcomeData& shot);
    std::vector<std::string> generateImprovementTips(int playerId);

private:
    // Background analysis thread
    void analysisLoop();
    void updatePlayerModels();
    void schedulePlayerUpdate(int playerId);
    
    // Feature extraction for ML models
    std::vector<float> extractShotFeatures(const DataCollectionEngine::ShotOutcomeData& shot);
    std::vector<float> extractPositionalFeatures(const cv::Point2f& position, 
                                                const std::vector<Ball>& balls);
    std::vector<float> extractGameStateFeatures(const GameState& gameState);
    
    // Utility functions
    float calculateShotComplexity(const DataCollectionEngine::ShotOutcomeData& shot);
    float estimatePlayerSkillLevel(int playerId);
    std::string generateShotStrategy(int playerId, const GameState& gameState);
    
    // Performance monitoring
    struct AnalysisMetrics {
        std::atomic<uint64_t> predictionsGenerated{0};
        std::atomic<uint64_t> modelsUpdated{0};
        std::atomic<double> avgPredictionTime{0.0};
        std::atomic<double> avgAnalysisTime{0.0};
    };
    
    AnalysisMetrics metrics_;
    
public:
    AnalysisMetrics getAnalysisMetrics() const { 
        AnalysisMetrics copy;
        copy.shotsAnalyzed = metrics_.shotsAnalyzed.load();
        copy.predictionsGenerated = metrics_.predictionsGenerated.load(); 
        copy.patternsRecognized = metrics_.patternsRecognized.load();
        copy.modelsUpdated = metrics_.modelsUpdated.load();
        copy.avgPredictionTime = metrics_.avgPredictionTime.load();
        copy.avgAnalysisTime = metrics_.avgAnalysisTime.load();
        return copy;
    }
    void logAnalysisReport();
};

/**
 * Factory for creating shot analysis engines
 */
class ShotAnalysisFactory {
public:
    static std::unique_ptr<ShotAnalysisEngine> createRealTime(DataCollectionEngine* dataCollection);
    static std::unique_ptr<ShotAnalysisEngine> createAdvanced(DataCollectionEngine* dataCollection);
    static std::unique_ptr<ShotAnalysisEngine> createLightweight(DataCollectionEngine* dataCollection);
};

} // namespace learning
} // namespace ai
} // namespace pv