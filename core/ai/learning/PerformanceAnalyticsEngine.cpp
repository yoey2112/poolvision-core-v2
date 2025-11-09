#include "PerformanceAnalyticsEngine.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <random>
#include <iostream>

namespace pv {
namespace ai {
namespace learning {

// StatisticalAnalyzer Implementation
PerformanceAnalyticsEngine::TrendAnalysis::TrendType PerformanceAnalyticsEngine::StatisticalAnalyzer::detectTrend(
    const std::vector<float>& data) {
    
    if (data.size() < 3) return TrendAnalysis::Stable;
    
    // Calculate linear regression slope
    float n = static_cast<float>(data.size());
    float sumX = n * (n - 1) / 2;
    float sumY = std::accumulate(data.begin(), data.end(), 0.0f);
    float sumXY = 0, sumX2 = 0;
    
    for (size_t i = 0; i < data.size(); ++i) {
        sumXY += i * data[i];
        sumX2 += i * i;
    }
    
    float slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    
    // Calculate variance to detect consistency
    float mean = sumY / n;
    float variance = 0;
    for (float value : data) {
        variance += std::pow(value - mean, 2);
    }
    variance /= n;
    
    // Classify trend
    float trendThreshold = 0.01f;
    float varianceThreshold = 0.1f;
    
    if (std::abs(slope) < trendThreshold && variance < varianceThreshold) {
        return TrendAnalysis::Stable;
    } else if (slope > trendThreshold) {
        return variance > varianceThreshold ? TrendAnalysis::Inconsistent : TrendAnalysis::Improving;
    } else if (slope < -trendThreshold) {
        return TrendAnalysis::Declining;
    } else if (variance > varianceThreshold * 2) {
        // Check for breakthrough pattern (sudden improvement)
        if (data.size() >= 5) {
            float recentMean = std::accumulate(data.end() - 3, data.end(), 0.0f) / 3;
            float historicMean = std::accumulate(data.begin(), data.end() - 3, 0.0f) / (data.size() - 3);
            if (recentMean > historicMean + 0.2f) {
                return TrendAnalysis::Breakthrough;
            }
        }
        return TrendAnalysis::Inconsistent;
    }
    
    return TrendAnalysis::Stable;
}

float PerformanceAnalyticsEngine::StatisticalAnalyzer::calculateTrendStrength(const std::vector<float>& data) {
    if (data.size() < 2) return 0.0f;
    
    // Calculate R-squared for linear regression
    float n = static_cast<float>(data.size());
    float sumX = n * (n - 1) / 2;
    float sumY = std::accumulate(data.begin(), data.end(), 0.0f);
    float sumXY = 0, sumX2 = 0, sumY2 = 0;
    
    for (size_t i = 0; i < data.size(); ++i) {
        sumXY += i * data[i];
        sumX2 += i * i;
        sumY2 += data[i] * data[i];
    }
    
    float numerator = n * sumXY - sumX * sumY;
    float denominator = std::sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    if (denominator == 0) return 0.0f;
    
    float correlation = numerator / denominator;
    return std::abs(correlation); // R-squared is correlation squared, but we want strength
}

float PerformanceAnalyticsEngine::StatisticalAnalyzer::calculateTrendConfidence(const std::vector<float>& data) {
    if (data.size() < 3) return 0.0f;
    
    float strength = calculateTrendStrength(data);
    float sampleSizeBonus = std::min(1.0f, static_cast<float>(data.size()) / 20.0f);
    
    return strength * sampleSizeBonus;
}

float PerformanceAnalyticsEngine::StatisticalAnalyzer::predictNextValue(const std::vector<float>& data) {
    if (data.empty()) return 0.5f;
    if (data.size() == 1) return data[0];
    
    // Exponential smoothing for prediction
    float alpha = 0.3f; // Smoothing parameter
    float prediction = data[0];
    
    for (size_t i = 1; i < data.size(); ++i) {
        prediction = alpha * data[i] + (1 - alpha) * prediction;
    }
    
    // Add trend component if strong trend exists
    if (data.size() >= 5) {
        std::vector<float> recent(data.end() - 5, data.end());
        float slope = 0;
        
        // Simple slope calculation
        for (size_t i = 1; i < recent.size(); ++i) {
            slope += (recent[i] - recent[i-1]);
        }
        slope /= (recent.size() - 1);
        
        prediction += slope; // Add trend projection
    }
    
    return std::clamp(prediction, 0.0f, 1.0f);
}

bool PerformanceAnalyticsEngine::StatisticalAnalyzer::isBreakthroughPerformance(
    float current, const std::vector<float>& history) {
    
    if (history.size() < 5) return false;
    
    float mean = std::accumulate(history.begin(), history.end(), 0.0f) / history.size();
    float variance = 0;
    for (float value : history) {
        variance += std::pow(value - mean, 2);
    }
    variance /= history.size();
    float stddev = std::sqrt(variance);
    
    // Breakthrough if current performance is 2+ standard deviations above mean
    return current > (mean + 2 * stddev) && current > 0.8f;
}

// VisualizationEngine Implementation
cv::Mat PerformanceAnalyticsEngine::VisualizationEngine::generatePerformanceTrendChart(
    const std::vector<float>& data, const std::string& title) {
    
    cv::Mat chart(400, 600, CV_8UC3, cv::Scalar(255, 255, 255));
    
    if (data.empty()) {
        cv::putText(chart, "No data available", cv::Point(150, 200), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
        return chart;
    }
    
    // Chart area
    cv::Rect chartArea(50, 50, 500, 300);
    cv::rectangle(chart, chartArea, cv::Scalar(200, 200, 200), 1);
    
    // Find data range
    auto minMax = std::minmax_element(data.begin(), data.end());
    float minVal = *minMax.first;
    float maxVal = *minMax.second;
    float range = maxVal - minVal;
    if (range == 0) range = 1.0f;
    
    // Convert data points to chart coordinates
    std::vector<cv::Point2f> points;
    for (size_t i = 0; i < data.size(); ++i) {
        float x = chartArea.x + (static_cast<float>(i) / (data.size() - 1)) * chartArea.width;
        float y = chartArea.y + chartArea.height - ((data[i] - minVal) / range) * chartArea.height;
        points.push_back(cv::Point2f(x, y));
    }
    
    // Draw trend line
    drawChart(chart, points, cv::Scalar(0, 100, 255), 2);
    
    // Draw data points
    for (const auto& point : points) {
        cv::circle(chart, point, 4, cv::Scalar(255, 0, 0), -1);
    }
    
    // Add title
    cv::putText(chart, title, cv::Point(50, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    
    return chart;
}

cv::Mat PerformanceAnalyticsEngine::VisualizationEngine::generateSkillRadarChart(
    const std::map<std::string, float>& skills) {
    
    cv::Mat chart(400, 400, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Point center(200, 200);
    int radius = 150;
    
    if (skills.empty()) return chart;
    
    // Draw radar grid
    for (int r = radius / 5; r <= radius; r += radius / 5) {
        cv::circle(chart, center, r, cv::Scalar(200, 200, 200), 1);
    }
    
    // Calculate angles for each skill
    std::vector<std::pair<std::string, float>> skillVector(skills.begin(), skills.end());
    float angleStep = 2 * M_PI / skillVector.size();
    
    // Draw radar spokes
    for (size_t i = 0; i < skillVector.size(); ++i) {
        float angle = i * angleStep;
        cv::Point endPoint(
            center.x + radius * std::cos(angle - M_PI / 2),
            center.y + radius * std::sin(angle - M_PI / 2)
        );
        cv::line(chart, center, endPoint, cv::Scalar(200, 200, 200), 1);
        
        // Add skill labels
        cv::Point labelPoint(
            center.x + (radius + 20) * std::cos(angle - M_PI / 2),
            center.y + (radius + 20) * std::sin(angle - M_PI / 2)
        );
        cv::putText(chart, skillVector[i].first.substr(0, 8), labelPoint,
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }
    
    // Draw skill polygon
    std::vector<cv::Point> polygon;
    for (size_t i = 0; i < skillVector.size(); ++i) {
        float angle = i * angleStep;
        float skillRadius = skillVector[i].second * radius;
        cv::Point point(
            center.x + skillRadius * std::cos(angle - M_PI / 2),
            center.y + skillRadius * std::sin(angle - M_PI / 2)
        );
        polygon.push_back(point);
    }
    
    // Fill polygon with transparency
    cv::Mat overlay = chart.clone();
    cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{polygon}, cv::Scalar(100, 150, 255));
    cv::addWeighted(chart, 0.7, overlay, 0.3, 0, chart);
    
    // Draw polygon outline
    for (size_t i = 0; i < polygon.size(); ++i) {
        cv::line(chart, polygon[i], polygon[(i + 1) % polygon.size()], cv::Scalar(0, 100, 255), 2);
    }
    
    return chart;
}

std::string PerformanceAnalyticsEngine::VisualizationEngine::generateTextualReport(
    const PerformanceAnalyticsEngine::SessionReport& report) {
    
    std::stringstream ss;
    
    ss << "=== Session Report ===" << std::endl;
    ss << "Player ID: " << report.playerId << std::endl;
    
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(
        report.sessionEnd - report.sessionStart);
    ss << "Duration: " << duration.count() << " minutes" << std::endl;
    
    ss << "\n--- Performance ---" << std::endl;
    ss << "Total Shots: " << report.performance.totalShots << std::endl;
    ss << "Successful Shots: " << report.performance.successfulShots << std::endl;
    
    if (report.performance.totalShots > 0) {
        float successRate = static_cast<float>(report.performance.successfulShots) / 
                           report.performance.totalShots * 100.0f;
        ss << "Success Rate: " << std::fixed << std::setprecision(1) << successRate << "%" << std::endl;
    }
    
    ss << "Session Rating: " << std::fixed << std::setprecision(2) << report.performance.sessionRating << std::endl;
    ss << "Best Streak: " << report.performance.bestStreak << std::endl;
    
    if (!report.insights.empty()) {
        ss << "\n--- Key Insights ---" << std::endl;
        for (const auto& insight : report.insights) {
            ss << "- " << insight.description << std::endl;
        }
    }
    
    if (!report.recommendations.empty()) {
        ss << "\n--- Recommendations ---" << std::endl;
        for (const auto& rec : report.recommendations) {
            ss << "- " << rec << std::endl;
        }
    }
    
    return ss.str();
}

void PerformanceAnalyticsEngine::VisualizationEngine::drawChart(
    cv::Mat& image, const std::vector<cv::Point2f>& points, const cv::Scalar& color, int thickness) {
    
    if (points.size() < 2) return;
    
    for (size_t i = 1; i < points.size(); ++i) {
        cv::line(image, points[i-1], points[i], color, thickness);
    }
}

// InsightGenerator Implementation
std::vector<PerformanceAnalyticsEngine::PerformanceInsight> 
PerformanceAnalyticsEngine::InsightGenerator::generateInsights(const PerformanceMetrics& metrics) {
    
    std::vector<PerformanceInsight> insights;
    
    // Skill balance analysis
    float maxSkill = std::max({metrics.advanced.skillProgression, 
                              metrics.basicStats.successRate,
                              metrics.advanced.consistencyIndex});
    float minSkill = std::min({metrics.advanced.skillProgression,
                              metrics.basicStats.successRate,
                              metrics.advanced.consistencyIndex});
    
    if (maxSkill - minSkill > 0.3f) {
        PerformanceInsight insight;
        insight.type = PerformanceInsight::Opportunity;
        insight.title = "Skill Imbalance Detected";
        insight.description = "Your skills are developing unevenly. Focus on weaker areas for balanced improvement.";
        insight.importance = 0.7f;
        insight.timestamp = std::chrono::steady_clock::now();
        insight.actionItems.push_back("Practice your weakest skill areas");
        insight.actionItems.push_back("Balance training between different shot types");
        insights.push_back(insight);
    }
    
    // Consistency analysis
    if (metrics.advanced.consistencyIndex < 0.6f && metrics.basicStats.successRate > 0.7f) {
        PerformanceInsight insight;
        insight.type = PerformanceInsight::Weakness;
        insight.title = "Consistency Needs Improvement";
        insight.description = "You can make good shots but lack consistency. Focus on routine and fundamentals.";
        insight.importance = 0.8f;
        insight.timestamp = std::chrono::steady_clock::now();
        insight.actionItems.push_back("Develop a consistent pre-shot routine");
        insight.actionItems.push_back("Practice fundamental stance and grip");
        insights.push_back(insight);
    }
    
    // Performance trend analysis
    if (metrics.temporal.sessionRatings.size() >= 5) {
        float recentAvg = std::accumulate(metrics.temporal.sessionRatings.end() - 3, 
                                        metrics.temporal.sessionRatings.end(), 0.0f) / 3;
        float overallAvg = std::accumulate(metrics.temporal.sessionRatings.begin(),
                                         metrics.temporal.sessionRatings.end(), 0.0f) / 
                          metrics.temporal.sessionRatings.size();
        
        if (recentAvg > overallAvg + 0.1f) {
            PerformanceInsight insight;
            insight.type = PerformanceInsight::Achievement;
            insight.title = "Improving Performance Trend";
            insight.description = "Your recent sessions show significant improvement. Keep up the good work!";
            insight.importance = 0.9f;
            insight.timestamp = std::chrono::steady_clock::now();
            insight.actionItems.push_back("Continue current practice routine");
            insight.actionItems.push_back("Challenge yourself with harder shots");
            insights.push_back(insight);
        }
    }
    
    return insights;
}

std::vector<std::string> PerformanceAnalyticsEngine::InsightGenerator::detectAchievements(
    const PerformanceMetrics& current, const PerformanceMetrics& previous) {
    
    std::vector<std::string> achievements;
    
    // Success rate milestones
    if (current.basicStats.successRate >= 0.8f && previous.basicStats.successRate < 0.8f) {
        achievements.push_back("80% Success Rate Achieved!");
    }
    if (current.basicStats.successRate >= 0.9f && previous.basicStats.successRate < 0.9f) {
        achievements.push_back("90% Success Rate Achieved!");
    }
    
    // Streak achievements
    if (current.temporal.bestStreak >= 10 && previous.temporal.bestStreak < 10) {
        achievements.push_back("10-Shot Streak Achieved!");
    }
    if (current.temporal.bestStreak >= 20 && previous.temporal.bestStreak < 20) {
        achievements.push_back("20-Shot Streak Achieved!");
    }
    
    // Consistency achievements
    if (current.advanced.consistencyIndex >= 0.8f && previous.advanced.consistencyIndex < 0.8f) {
        achievements.push_back("High Consistency Achieved!");
    }
    
    return achievements;
}

// PerformanceAnalyticsEngine Implementation
PerformanceAnalyticsEngine::PerformanceAnalyticsEngine(DataCollectionEngine* dataEngine,
                                                       ShotAnalysisEngine* analysisEngine,
                                                       AdaptiveCoachingEngine* coachingEngine)
    : dataEngine_(dataEngine), analysisEngine_(analysisEngine), coachingEngine_(coachingEngine) {
    
    statisticalAnalyzer_ = std::make_unique<StatisticalAnalyzer>();
    visualizationEngine_ = std::make_unique<VisualizationEngine>();
    insightGenerator_ = std::make_unique<InsightGenerator>();
    
    std::cout << "Performance Analytics Engine initialized" << std::endl;
}

PerformanceAnalyticsEngine::~PerformanceAnalyticsEngine() {
    stopAnalytics();
}

void PerformanceAnalyticsEngine::startAnalytics() {
    if (analyticsActive_.load()) return;
    
    analyticsActive_ = true;
    analyticsThread_ = std::thread(&PerformanceAnalyticsEngine::analyticsLoop, this);
    
    std::cout << "Performance Analytics Engine started" << std::endl;
}

void PerformanceAnalyticsEngine::stopAnalytics() {
    if (!analyticsActive_.load()) return;
    
    analyticsActive_ = false;
    analyticsCondition_.notify_all();
    
    if (analyticsThread_.joinable()) {
        analyticsThread_.join();
    }
    
    std::cout << "Performance Analytics Engine stopped" << std::endl;
}

void PerformanceAnalyticsEngine::updatePlayerPerformance(int playerId, 
                                                        const DataCollectionEngine::ShotOutcomeData& shot) {
    // Add to processing queue
    std::lock_guard<std::mutex> lock(queueMutex_);
    updateQueue_.push(std::make_pair(playerId, shot));
    analyticsCondition_.notify_one();
}

PerformanceAnalyticsEngine::PerformanceMetrics PerformanceAnalyticsEngine::getPlayerMetrics(int playerId) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    auto it = playerMetrics_.find(playerId);
    if (it != playerMetrics_.end()) {
        return it->second;
    }
    
    // Return empty metrics if player not found
    return PerformanceMetrics{};
}

PerformanceAnalyticsEngine::TrendAnalysis PerformanceAnalyticsEngine::analyzePerformanceTrends(int playerId) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    TrendAnalysis analysis;
    auto metrics = getPlayerMetrics(playerId);
    
    if (metrics.temporal.sessionRatings.empty()) {
        analysis.overallTrend = TrendAnalysis::Stable;
        analysis.trendStrength = 0.0f;
        analysis.trendConfidence = 0.0f;
        return analysis;
    }
    
    // Analyze overall trend
    analysis.overallTrend = statisticalAnalyzer_->detectTrend(metrics.temporal.sessionRatings);
    analysis.trendStrength = statisticalAnalyzer_->calculateTrendStrength(metrics.temporal.sessionRatings);
    analysis.trendConfidence = statisticalAnalyzer_->calculateTrendConfidence(metrics.temporal.sessionRatings);
    
    // Generate prediction
    analysis.prediction.predictedRating = statisticalAnalyzer_->predictNextValue(metrics.temporal.sessionRatings);
    analysis.prediction.confidenceInterval = 0.1f * (1.0f - analysis.trendConfidence);
    
    // Estimate days to improvement
    if (analysis.overallTrend == TrendAnalysis::Improving && analysis.trendStrength > 0.3f) {
        analysis.prediction.daysToImprovement = static_cast<int>(10.0f / analysis.trendStrength);
    } else {
        analysis.prediction.daysToImprovement = 30; // Conservative estimate
    }
    
    // Generate reasoning
    std::stringstream reasoning;
    switch (analysis.overallTrend) {
        case TrendAnalysis::Improving:
            reasoning << "Showing consistent improvement";
            break;
        case TrendAnalysis::Declining:
            reasoning << "Performance declining, may need coaching adjustment";
            break;
        case TrendAnalysis::Breakthrough:
            reasoning << "Breakthrough performance detected!";
            break;
        case TrendAnalysis::Inconsistent:
            reasoning << "Performance varies significantly";
            break;
        default:
            reasoning << "Stable performance";
            break;
    }
    analysis.prediction.reasoning = reasoning.str();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double analysisTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    // Update metrics
    double currentAvg = metrics_.avgAnalysisTime.load();
    metrics_.avgAnalysisTime.store(currentAvg * 0.9 + analysisTime * 0.1);
    
    return analysis;
}

std::vector<PerformanceAnalyticsEngine::PerformanceInsight> 
PerformanceAnalyticsEngine::generatePerformanceInsights(int playerId) {
    auto metrics = getPlayerMetrics(playerId);
    auto insights = insightGenerator_->generateInsights(metrics);
    
    metrics_.insightsGenerated.fetch_add(static_cast<int>(insights.size()));
    
    return insights;
}

void PerformanceAnalyticsEngine::startPerformanceSession(int playerId) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    sessionStarts_[playerId] = std::chrono::steady_clock::now();
    
    // Initialize metrics if needed
    if (playerMetrics_.find(playerId) == playerMetrics_.end()) {
        initializePlayerMetrics(playerId);
    }
    
    metrics_.sessionsAnalyzed.fetch_add(1);
}

void PerformanceAnalyticsEngine::endPerformanceSession(int playerId) {
    auto report = generateSessionReport(playerId);
    
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    // Store session report
    sessionHistory_[playerId].push_back(report);
    if (sessionHistory_[playerId].size() > 50) {
        sessionHistory_[playerId].erase(sessionHistory_[playerId].begin());
    }
    
    // Remove session start time
    sessionStarts_.erase(playerId);
}

PerformanceAnalyticsEngine::SessionReport PerformanceAnalyticsEngine::generateSessionReport(int playerId) {
    SessionReport report;
    report.playerId = playerId;
    
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    auto startIt = sessionStarts_.find(playerId);
    if (startIt != sessionStarts_.end()) {
        report.sessionStart = startIt->second;
        report.sessionEnd = std::chrono::steady_clock::now();
    }
    
    auto metricsIt = playerMetrics_.find(playerId);
    if (metricsIt != playerMetrics_.end()) {
        const auto& metrics = metricsIt->second;
        
        // Calculate session performance based on recent data
        // This is simplified - in practice, you'd track session-specific data
        report.performance.totalShots = metrics.basicStats.totalShots;
        report.performance.successfulShots = metrics.basicStats.successfulShots;
        report.performance.sessionRating = metrics.basicStats.successRate;
        report.performance.bestStreak = metrics.temporal.bestStreak;
        report.performance.averageConfidence = 0.7f; // Placeholder
        
        // Generate insights for this session
        report.insights = insightGenerator_->generateInsights(metrics);
        
        // Generate recommendations
        report.recommendations.push_back("Focus on consistency");
        report.recommendations.push_back("Practice challenging shots");
        
        // Set comparative data
        if (!metrics.temporal.sessionRatings.empty()) {
            float recentAvg = std::accumulate(metrics.temporal.sessionRatings.begin(),
                                            metrics.temporal.sessionRatings.end(), 0.0f) / 
                             metrics.temporal.sessionRatings.size();
            report.comparative.vsRecentAverage = report.performance.sessionRating - recentAvg;
        }
        report.comparative.vsPersonalBest = report.performance.sessionRating - metrics.advanced.peakPerformanceLevel;
        
        // Determine performance category
        if (report.performance.sessionRating >= 0.9f) {
            report.comparative.performanceCategory = "excellent";
        } else if (report.performance.sessionRating >= 0.7f) {
            report.comparative.performanceCategory = "good";
        } else if (report.performance.sessionRating >= 0.5f) {
            report.comparative.performanceCategory = "average";
        } else {
            report.comparative.performanceCategory = "below average";
        }
    }
    
    return report;
}

PerformanceAnalyticsEngine::PerformancePrediction 
PerformanceAnalyticsEngine::generatePerformancePrediction(int playerId) {
    PerformancePrediction prediction;
    auto metrics = getPlayerMetrics(playerId);
    
    if (!metrics.temporal.sessionRatings.empty()) {
        // Short-term prediction
        prediction.shortTerm.expectedSuccessRate = 
            statisticalAnalyzer_->predictNextValue(metrics.temporal.sessionRatings);
        prediction.shortTerm.confidenceLevel = 
            statisticalAnalyzer_->calculateTrendConfidence(metrics.temporal.sessionRatings);
        prediction.shortTerm.factors.push_back("Recent performance trend");
        prediction.shortTerm.factors.push_back("Skill consistency level");
        
        // Medium-term prediction
        prediction.mediumTerm.skillImprovement = metrics.advanced.learningVelocity * 7; // Week projection
        prediction.mediumTerm.practiceRecommendation = 
            (1.0f - metrics.advanced.consistencyIndex) * 10; // Hours per week
        prediction.mediumTerm.focusAreas.push_back("Consistency training");
        
        // Long-term prediction
        prediction.longTerm.skillCeilingProgress = 
            metrics.basicStats.successRate / (metrics.comparative.skillCeiling + 0.01f);
        prediction.longTerm.masteryCurve = calculateLearningVelocity(playerId);
        prediction.longTerm.milestones.push_back("80% success rate");
        prediction.longTerm.milestones.push_back("Advanced shot mastery");
    }
    
    metrics_.predictionsGenerated.fetch_add(1);
    
    return prediction;
}

float PerformanceAnalyticsEngine::predictSessionRating(int playerId) {
    auto metrics = getPlayerMetrics(playerId);
    
    if (metrics.temporal.sessionRatings.empty()) {
        return 0.5f; // Default prediction
    }
    
    return statisticalAnalyzer_->predictNextValue(metrics.temporal.sessionRatings);
}

cv::Mat PerformanceAnalyticsEngine::generatePerformanceChart(int playerId, const std::string& chartType) {
    auto metrics = getPlayerMetrics(playerId);
    
    if (chartType == "trend") {
        return visualizationEngine_->generatePerformanceTrendChart(
            metrics.temporal.sessionRatings, "Performance Trend");
    } else if (chartType == "skills") {
        std::map<std::string, float> skills = {
            {"Accuracy", metrics.basicStats.successRate},
            {"Consistency", metrics.advanced.consistencyIndex},
            {"Strategy", metrics.advanced.adaptabilityScore},
            {"Pressure", metrics.advanced.clutchPerformance}
        };
        return visualizationEngine_->generateSkillRadarChart(skills);
    }
    
    // Default: return empty chart
    cv::Mat chart(400, 600, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::putText(chart, "Chart type not supported", cv::Point(150, 200),
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    
    metrics_.chartsGenerated.fetch_add(1);
    
    return chart;
}

std::string PerformanceAnalyticsEngine::generatePerformanceReport(int playerId) {
    auto metrics = getPlayerMetrics(playerId);
    auto trends = analyzePerformanceTrends(playerId);
    
    std::stringstream report;
    
    report << "=== Performance Analytics Report ===" << std::endl;
    report << "Player ID: " << playerId << std::endl;
    
    report << "\n--- Current Performance ---" << std::endl;
    report << "Success Rate: " << std::fixed << std::setprecision(1) 
           << (metrics.basicStats.successRate * 100) << "%" << std::endl;
    report << "Total Shots: " << metrics.basicStats.totalShots << std::endl;
    report << "Consistency Index: " << std::setprecision(2) << metrics.advanced.consistencyIndex << std::endl;
    
    report << "\n--- Trend Analysis ---" << std::endl;
    report << "Overall Trend: ";
    switch (trends.overallTrend) {
        case TrendAnalysis::Improving: report << "Improving"; break;
        case TrendAnalysis::Declining: report << "Declining"; break;
        case TrendAnalysis::Breakthrough: report << "Breakthrough"; break;
        case TrendAnalysis::Inconsistent: report << "Inconsistent"; break;
        default: report << "Stable"; break;
    }
    report << std::endl;
    report << "Trend Strength: " << trends.trendStrength << std::endl;
    report << "Predicted Next Rating: " << trends.prediction.predictedRating << std::endl;
    
    // Add insights
    auto insights = generatePerformanceInsights(playerId);
    if (!insights.empty()) {
        report << "\n--- Key Insights ---" << std::endl;
        for (const auto& insight : insights) {
            report << "- " << insight.description << std::endl;
        }
    }
    
    return report.str();
}

float PerformanceAnalyticsEngine::calculateRealTimePerformanceScore(int playerId) {
    auto metrics = getPlayerMetrics(playerId);
    
    // Weighted combination of key metrics
    float score = metrics.basicStats.successRate * 0.4f +
                  metrics.advanced.consistencyIndex * 0.3f +
                  metrics.advanced.adaptabilityScore * 0.2f +
                  metrics.advanced.learningVelocity * 0.1f;
    
    return std::clamp(score, 0.0f, 1.0f);
}

void PerformanceAnalyticsEngine::analyticsLoop() {
    while (analyticsActive_.load()) {
        std::unique_lock<std::mutex> lock(queueMutex_);
        
        analyticsCondition_.wait_for(lock, std::chrono::seconds(updateFrequency_), [this] {
            return !analyticsActive_.load() || !updateQueue_.empty();
        });
        
        if (!analyticsActive_.load()) break;
        
        processUpdateQueue();
        lock.unlock();
        
        updateGlobalStatistics();
        generateScheduledInsights();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void PerformanceAnalyticsEngine::processUpdateQueue() {
    while (!updateQueue_.empty()) {
        auto [playerId, shot] = updateQueue_.front();
        updateQueue_.pop();
        
        recalculatePlayerAnalytics(playerId);
    }
}

void PerformanceAnalyticsEngine::recalculatePlayerAnalytics(int playerId) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    
    if (playerMetrics_.find(playerId) == playerMetrics_.end()) {
        initializePlayerMetrics(playerId);
    }
    
    auto& metrics = playerMetrics_[playerId];
    
    // Get recent shot data
    auto recentShots = dataEngine_->getPlayerShotHistory(playerId, 100);
    
    // Update basic stats
    metrics.basicStats.totalShots = static_cast<int>(recentShots.size());
    metrics.basicStats.successfulShots = 0;
    float totalDifficulty = 0, totalSpeed = 0;
    
    for (const auto& shot : recentShots) {
        if (shot.successful) metrics.basicStats.successfulShots++;
        totalDifficulty += shot.shotDifficulty;
        totalSpeed += shot.shotSpeed;
    }
    
    if (!recentShots.empty()) {
        metrics.basicStats.successRate = static_cast<float>(metrics.basicStats.successfulShots) / 
                                        recentShots.size();
        metrics.basicStats.averageDifficulty = totalDifficulty / recentShots.size();
        metrics.basicStats.averageSpeed = totalSpeed / recentShots.size();
    }
    
    // Update advanced analytics
    updateAdvancedAnalytics(metrics, playerId);
    updateTemporalAnalytics(metrics, playerId);
    updateComparativeAnalytics(metrics, playerId);
}

void PerformanceAnalyticsEngine::updateAdvancedAnalytics(PerformanceMetrics& metrics, int playerId) {
    metrics.advanced.skillProgression = calculateSkillProgression(playerId);
    metrics.advanced.consistencyIndex = calculateConsistencyIndex(metrics.temporal.sessionRatings);
    metrics.advanced.clutchPerformance = calculateClutchPerformance(playerId);
    metrics.advanced.adaptabilityScore = calculateAdaptabilityScore(playerId);
    metrics.advanced.learningVelocity = calculateLearningVelocity(playerId);
    
    // Update peak performance
    if (metrics.basicStats.successRate > metrics.advanced.peakPerformanceLevel) {
        metrics.advanced.peakPerformanceLevel = metrics.basicStats.successRate;
    }
}

void PerformanceAnalyticsEngine::initializePlayerMetrics(int playerId) {
    PerformanceMetrics metrics;
    
    // Initialize all values to defaults
    metrics.basicStats = {};
    metrics.advanced = {};
    metrics.temporal = {};
    metrics.comparative = {};
    
    playerMetrics_[playerId] = metrics;
    metrics_.playersTracked.fetch_add(1);
}

float PerformanceAnalyticsEngine::calculateSkillProgression(int playerId) {
    auto metrics = getPlayerMetrics(playerId);
    
    if (metrics.temporal.sessionRatings.size() < 5) return 0.0f;
    
    // Calculate improvement over time
    auto recent = std::vector<float>(metrics.temporal.sessionRatings.end() - 3,
                                   metrics.temporal.sessionRatings.end());
    auto earlier = std::vector<float>(metrics.temporal.sessionRatings.begin(),
                                    metrics.temporal.sessionRatings.begin() + 3);
    
    float recentAvg = std::accumulate(recent.begin(), recent.end(), 0.0f) / recent.size();
    float earlierAvg = std::accumulate(earlier.begin(), earlier.end(), 0.0f) / earlier.size();
    
    return std::clamp(recentAvg - earlierAvg, -1.0f, 1.0f);
}

float PerformanceAnalyticsEngine::calculateConsistencyIndex(const std::vector<float>& performances) {
    if (performances.size() < 3) return 0.5f;
    
    float mean = std::accumulate(performances.begin(), performances.end(), 0.0f) / performances.size();
    float variance = 0;
    
    for (float perf : performances) {
        variance += std::pow(perf - mean, 2);
    }
    variance /= performances.size();
    
    // Convert variance to consistency (lower variance = higher consistency)
    return std::clamp(1.0f - variance, 0.0f, 1.0f);
}

float PerformanceAnalyticsEngine::calculateClutchPerformance(int playerId) {
    // Simplified: return average performance for now
    auto metrics = getPlayerMetrics(playerId);
    return metrics.basicStats.successRate;
}

float PerformanceAnalyticsEngine::calculateAdaptabilityScore(int playerId) {
    // Simplified: based on shot type diversity
    auto shots = dataEngine_->getPlayerShotHistory(playerId, 50);
    if (shots.empty()) return 0.5f;
    
    std::set<DataCollectionEngine::ShotOutcomeData::ShotType> uniqueTypes;
    for (const auto& shot : shots) {
        uniqueTypes.insert(shot.shotType);
    }
    
    return static_cast<float>(uniqueTypes.size()) / 7.0f; // Assuming 7 shot types
}

float PerformanceAnalyticsEngine::calculateLearningVelocity(int playerId) {
    auto metrics = getPlayerMetrics(playerId);
    
    if (metrics.temporal.sessionRatings.size() < 10) return 0.0f;
    
    // Calculate rate of improvement over time
    return statisticalAnalyzer_->calculateTrendStrength(metrics.temporal.sessionRatings) * 
           (metrics.temporal.sessionRatings.back() > metrics.temporal.sessionRatings.front() ? 1.0f : -1.0f);
}

void PerformanceAnalyticsEngine::logAnalyticsReport() {
    auto analyticsMetrics = getAnalyticsMetrics();
    
    std::cout << "\n=== Performance Analytics Engine Report ===" << std::endl;
    std::cout << "Players tracked: " << analyticsMetrics.playersTracked.load() << std::endl;
    std::cout << "Sessions analyzed: " << analyticsMetrics.sessionsAnalyzed.load() << std::endl;
    std::cout << "Insights generated: " << analyticsMetrics.insightsGenerated.load() << std::endl;
    std::cout << "Predictions generated: " << analyticsMetrics.predictionsGenerated.load() << std::endl;
    std::cout << "Charts generated: " << analyticsMetrics.chartsGenerated.load() << std::endl;
    std::cout << "Avg analysis time: " << std::fixed << std::setprecision(2) 
              << analyticsMetrics.avgAnalysisTime.load() << "ms" << std::endl;
    std::cout << "=============================================" << std::endl;
}

// Factory Implementation
std::unique_ptr<PerformanceAnalyticsEngine> PerformanceAnalyticsFactory::createRealTime(
    DataCollectionEngine* dataEngine, ShotAnalysisEngine* analysisEngine,
    AdaptiveCoachingEngine* coachingEngine) {
    
    auto engine = std::make_unique<PerformanceAnalyticsEngine>(dataEngine, analysisEngine, coachingEngine);
    engine->setUpdateFrequency(5); // Real-time updates every 5 seconds
    engine->setAnalyticsDepth(3);  // Comprehensive analysis
    return engine;
}

std::unique_ptr<PerformanceAnalyticsEngine> PerformanceAnalyticsFactory::createBatch(
    DataCollectionEngine* dataEngine, ShotAnalysisEngine* analysisEngine,
    AdaptiveCoachingEngine* coachingEngine) {
    
    auto engine = std::make_unique<PerformanceAnalyticsEngine>(dataEngine, analysisEngine, coachingEngine);
    engine->setUpdateFrequency(60); // Batch updates every minute
    engine->setAnalyticsDepth(2);   // Standard analysis
    return engine;
}

} // namespace learning
} // namespace ai
} // namespace pv