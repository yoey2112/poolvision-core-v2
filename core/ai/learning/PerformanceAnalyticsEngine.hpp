#ifndef PV_AI_LEARNING_PERFORMANCE_ANALYTICS_HPP
#define PV_AI_LEARNING_PERFORMANCE_ANALYTICS_HPP

#include "DataCollectionEngine.hpp"
#include "ShotAnalysisEngine.hpp"
#include "AdaptiveCoachingEngine.hpp"
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <opencv2/opencv.hpp>

namespace pv {
namespace ai {
namespace learning {

/**
 * Performance Analytics Engine
 * 
 * Provides comprehensive performance analysis and insights:
 * - Real-time performance tracking and trend analysis
 * - Statistical performance modeling and predictions
 * - Session-based analytics with historical comparisons
 * - Performance visualizations and reports
 * - Predictive insights for improvement planning
 */
class PerformanceAnalyticsEngine {
public:
    // Performance Metrics
    struct PerformanceMetrics {
        // Basic Statistics
        struct BasicStats {
            int totalShots;
            int successfulShots;
            float successRate;
            float averageAccuracy;
            float averageDifficulty;
            float averageSpeed;
            std::chrono::seconds totalPlayTime;
            std::chrono::time_point<std::chrono::steady_clock> lastPlayed;
        } basicStats;
        
        // Advanced Analytics
        struct AdvancedAnalytics {
            float skillProgression;     // Overall skill improvement rate
            float consistencyIndex;     // Performance consistency metric
            float clutchPerformance;    // Performance under pressure
            float adaptabilityScore;    // Ability to handle different situations
            float learningVelocity;     // Rate of skill acquisition
            float peakPerformanceLevel; // Highest sustainable performance
        } advanced;
        
        // Shot Type Analysis
        std::map<DataCollectionEngine::ShotOutcomeData::ShotType, struct {
            float successRate;
            float improvementRate;
            int attempts;
            float avgDifficulty;
            float preference;       // How often player chooses this shot type
        }> shotTypeAnalysis;
        
        // Time-based Analytics
        struct TemporalAnalytics {
            std::vector<float> sessionRatings;          // Last 20 sessions
            std::vector<float> weeklyAverages;          // Weekly performance averages
            std::vector<float> improvementMilestones;   // Skill progression points
            float bestStreak;                           // Longest successful shot streak
            float currentStreak;                        // Current streak
            std::map<int, float> hourlyPerformance;     // Performance by hour of day
        } temporal;
        
        // Comparative Analytics
        struct ComparativeAnalytics {
            float percentileRank;       // Performance percentile among all players
            float skillCeiling;         // Estimated maximum potential
            float improvementPotential; // Remaining room for growth
            std::vector<std::string> strengthAreas;
            std::vector<std::string> improvementAreas;
        } comparative;
    };
    
    // Trend Analysis Data
    struct TrendAnalysis {
        enum TrendType {
            Improving,
            Stable,
            Declining,
            Inconsistent,
            Breakthrough
        };
        
        TrendType overallTrend;
        float trendStrength;        // How strong the trend is (0-1)
        float trendConfidence;      // Statistical confidence (0-1)
        
        struct TrendPrediction {
            float predictedRating;      // Predicted performance in next session
            float confidenceInterval;  // Uncertainty in prediction
            int daysToImprovement;     // Estimated days to next skill level
            std::string reasoning;     // Explanation of prediction
        } prediction;
        
        // Skill-specific trends
        std::map<std::string, struct {
            TrendType trend;
            float rate;
            float significance;
        }> skillTrends;
    };
    
    // Performance Insights
    struct PerformanceInsight {
        enum Type {
            Achievement,    // Milestone reached
            Improvement,    // Skill improvement detected
            Weakness,       // Area needing attention
            Opportunity,    // Chance for growth
            Warning,        // Performance decline
            Strategy        // Strategic recommendation
        };
        
        Type type;
        std::string title;
        std::string description;
        float importance;           // 0-1 how important this insight is
        std::vector<std::string> actionItems;
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
        
        // Supporting data
        struct Evidence {
            std::vector<float> supportingData;
            std::string methodology;
            float statisticalSignificance;
        } evidence;
    };
    
    // Session Report
    struct SessionReport {
        int sessionId;
        int playerId;
        std::chrono::time_point<std::chrono::steady_clock> sessionStart;
        std::chrono::time_point<std::chrono::steady_clock> sessionEnd;
        
        // Session Performance
        struct SessionPerformance {
            int totalShots;
            int successfulShots;
            float sessionRating;
            float improvementFromLast;
            float bestStreak;
            float averageConfidence;
            float fatigueFactor;        // Performance decline due to fatigue
        } performance;
        
        // Session Insights
        std::vector<PerformanceInsight> insights;
        std::vector<std::string> achievements;
        std::vector<std::string> recommendations;
        
        // Comparative Data
        struct SessionComparative {
            float vsPersonalBest;
            float vsRecentAverage;
            float vsExpected;
            std::string performanceCategory; // "excellent", "good", "average", "below average"
        } comparative;
    };
    
    // Predictive Models
    struct PerformancePrediction {
        // Short-term predictions (next session)
        struct ShortTerm {
            float expectedSuccessRate;
            float confidenceLevel;
            std::vector<std::string> factors;  // Factors influencing prediction
        } shortTerm;
        
        // Medium-term predictions (next week)
        struct MediumTerm {
            float skillImprovement;
            float practiceRecommendation;   // Hours of practice recommended
            std::vector<std::string> focusAreas;
        } mediumTerm;
        
        // Long-term predictions (next month)
        struct LongTerm {
            float skillCeilingProgress;     // Progress toward skill ceiling
            float masteryCurve;             // Shape of learning curve
            std::vector<std::string> milestones;
        } longTerm;
    };

private:
    // Statistical Analysis Engine
    class StatisticalAnalyzer {
    public:
        // Trend detection
        TrendAnalysis::TrendType detectTrend(const std::vector<float>& data);
        float calculateTrendStrength(const std::vector<float>& data);
        float calculateTrendConfidence(const std::vector<float>& data);
        
        // Statistical modeling
        std::vector<float> performRegression(const std::vector<float>& data);
        float calculateCorrelation(const std::vector<float>& x, const std::vector<float>& y);
        float calculateVariance(const std::vector<float>& data);
        float calculateSkewness(const std::vector<float>& data);
        
        // Anomaly detection
        std::vector<int> detectAnomalies(const std::vector<float>& data);
        bool isBreakthroughPerformance(float current, const std::vector<float>& history);
        
        // Predictive modeling
        float predictNextValue(const std::vector<float>& data);
        float calculatePredictionConfidence(const std::vector<float>& data, float prediction);
    };
    
    // Visualization Engine (for generating performance charts)
    class VisualizationEngine {
    public:
        // Generate performance charts
        cv::Mat generatePerformanceTrendChart(const std::vector<float>& data,
                                            const std::string& title);
        cv::Mat generateSkillRadarChart(const std::map<std::string, float>& skills);
        cv::Mat generateShotTypeAnalysisChart(const PerformanceMetrics::shotTypeAnalysis& data);
        cv::Mat generateImprovementTrajectoryChart(const std::vector<float>& milestones);
        
        // Report generation
        std::string generateTextualReport(const SessionReport& report);
        std::string generatePerformanceSummary(const PerformanceMetrics& metrics);
        
    private:
        void drawChart(cv::Mat& image, const std::vector<cv::Point2f>& points,
                      const cv::Scalar& color, int thickness = 2);
        void addChartLabels(cv::Mat& image, const std::string& title,
                          const std::string& xLabel, const std::string& yLabel);
    };
    
    // Insight Generator
    class InsightGenerator {
    public:
        // Generate insights from metrics
        std::vector<PerformanceInsight> generateInsights(const PerformanceMetrics& metrics);
        std::vector<PerformanceInsight> generateSessionInsights(const SessionReport& report);
        
        // Achievement detection
        std::vector<std::string> detectAchievements(const PerformanceMetrics& current,
                                                   const PerformanceMetrics& previous);
        
        // Recommendation engine
        std::vector<std::string> generateRecommendations(const PerformanceMetrics& metrics,
                                                        const TrendAnalysis& trends);
    };

public:
    explicit PerformanceAnalyticsEngine(DataCollectionEngine* dataEngine,
                                       ShotAnalysisEngine* analysisEngine,
                                       AdaptiveCoachingEngine* coachingEngine);
    ~PerformanceAnalyticsEngine();
    
    // Core functionality
    void startAnalytics();
    void stopAnalytics();
    
    // Performance tracking
    void updatePlayerPerformance(int playerId, const DataCollectionEngine::ShotOutcomeData& shot);
    PerformanceMetrics getPlayerMetrics(int playerId);
    PerformanceMetrics getPlayerMetrics(int playerId, 
                                       const std::chrono::time_point<std::chrono::steady_clock>& since);
    
    // Trend analysis
    TrendAnalysis analyzePerformanceTrends(int playerId);
    TrendAnalysis analyzeSkillTrends(int playerId, const std::string& skillName);
    std::vector<PerformanceInsight> generatePerformanceInsights(int playerId);
    
    // Session analytics
    void startPerformanceSession(int playerId);
    void endPerformanceSession(int playerId);
    SessionReport generateSessionReport(int playerId);
    std::vector<SessionReport> getSessionHistory(int playerId, int maxSessions = 20);
    
    // Predictive analytics
    PerformancePrediction generatePerformancePrediction(int playerId);
    float predictSessionRating(int playerId);
    std::vector<std::string> predictImprovementOpportunities(int playerId);
    
    // Comparative analytics
    std::map<std::string, float> compareToBaseline(int playerId);
    std::map<std::string, float> compareToPlayerPool(int playerId);
    float calculateSkillCeiling(int playerId);
    
    // Visualization and reporting
    cv::Mat generatePerformanceChart(int playerId, const std::string& chartType);
    std::string generatePerformanceReport(int playerId);
    std::string generateImprovementPlan(int playerId);
    
    // Real-time analytics
    float calculateRealTimePerformanceScore(int playerId);
    std::vector<std::string> getRealTimeInsights(int playerId);
    bool detectPerformanceAnomaly(int playerId, float currentScore);
    
    // Configuration
    void setAnalyticsDepth(int depth);      // 1=basic, 2=advanced, 3=comprehensive
    void setUpdateFrequency(int seconds);   // How often to recalculate analytics
    void enablePredictiveModeling(bool enable);
    void setVisualizationQuality(int quality); // 1=low, 2=medium, 3=high
    
    // Data export and import
    std::string exportPlayerAnalytics(int playerId, const std::string& format = "json");
    bool importPlayerAnalytics(int playerId, const std::string& data);
    void exportGlobalAnalytics(const std::string& filename);
    
    // Metrics and monitoring
    struct AnalyticsMetrics {
        std::atomic<int> playersTracked{0};
        std::atomic<int> sessionsAnalyzed{0};
        std::atomic<int> insightsGenerated{0};
        std::atomic<int> predictionsGenerated{0};
        std::atomic<double> avgAnalysisTime{0.0};
        std::atomic<double> avgPredictionAccuracy{0.0};
        std::atomic<int> chartsGenerated{0};
    };
    
    AnalyticsMetrics getAnalyticsMetrics() const { return metrics_; }
    void logAnalyticsReport();

private:
    // Core components
    DataCollectionEngine* dataEngine_;
    ShotAnalysisEngine* analysisEngine_;
    AdaptiveCoachingEngine* coachingEngine_;
    
    std::unique_ptr<StatisticalAnalyzer> statisticalAnalyzer_;
    std::unique_ptr<VisualizationEngine> visualizationEngine_;
    std::unique_ptr<InsightGenerator> insightGenerator_;
    
    // Player data storage
    std::map<int, PerformanceMetrics> playerMetrics_;
    std::map<int, std::vector<SessionReport>> sessionHistory_;
    std::map<int, std::chrono::time_point<std::chrono::steady_clock>> sessionStarts_;
    
    // Threading and synchronization
    std::atomic<bool> analyticsActive_{false};
    std::thread analyticsThread_;
    std::mutex dataMutex_;
    std::condition_variable analyticsCondition_;
    
    // Processing queue
    std::queue<std::pair<int, DataCollectionEngine::ShotOutcomeData>> updateQueue_;
    std::mutex queueMutex_;
    
    // Configuration
    int analyticsDepth_ = 2;
    int updateFrequency_ = 30; // seconds
    bool predictiveModelingEnabled_ = true;
    int visualizationQuality_ = 2;
    
    // Metrics
    AnalyticsMetrics metrics_;
    
    // Background processing
    void analyticsLoop();
    void processUpdateQueue();
    void recalculatePlayerAnalytics(int playerId);
    void updateGlobalStatistics();
    void generateScheduledInsights();
    
    // Helper methods
    void initializePlayerMetrics(int playerId);
    void updateBasicStats(PerformanceMetrics& metrics, 
                         const DataCollectionEngine::ShotOutcomeData& shot);
    void updateAdvancedAnalytics(PerformanceMetrics& metrics, int playerId);
    void updateTemporalAnalytics(PerformanceMetrics& metrics, int playerId);
    void updateComparativeAnalytics(PerformanceMetrics& metrics, int playerId);
    
    float calculateSkillProgression(int playerId);
    float calculateConsistencyIndex(const std::vector<float>& performances);
    float calculateClutchPerformance(int playerId);
    float calculateAdaptabilityScore(int playerId);
    float calculateLearningVelocity(int playerId);
    
    // Statistical helpers
    float calculateMovingAverage(const std::vector<float>& data, int window);
    float calculateExponentialSmoothing(const std::vector<float>& data, float alpha);
    std::vector<float> applyMovingAverageFilter(const std::vector<float>& data, int window);
    
    // Database helpers
    void saveAnalyticsToDatabase(int playerId, const PerformanceMetrics& metrics);
    PerformanceMetrics loadAnalyticsFromDatabase(int playerId);
};

// Factory for creating analytics engines
class PerformanceAnalyticsFactory {
public:
    static std::unique_ptr<PerformanceAnalyticsEngine> createRealTime(
        DataCollectionEngine* dataEngine,
        ShotAnalysisEngine* analysisEngine,
        AdaptiveCoachingEngine* coachingEngine);
    
    static std::unique_ptr<PerformanceAnalyticsEngine> createBatch(
        DataCollectionEngine* dataEngine,
        ShotAnalysisEngine* analysisEngine,
        AdaptiveCoachingEngine* coachingEngine);
};

} // namespace learning
} // namespace ai
} // namespace pv

#endif // PV_AI_LEARNING_PERFORMANCE_ANALYTICS_HPP