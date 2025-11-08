#pragma once

#include <string>
#include <map>
#include <vector>
#include <functional>

namespace pv {

class Config {
public:
    struct ValidationRule {
        std::string key;
        bool required;
        std::function<bool(const Config&)> validate;
        std::string description;
    };

    struct ValidationError {
        std::string key;
        std::string message;
    };

    bool load(const std::string &path);
    std::string getString(const std::string &key, const std::string &def="") const;
    double getDouble(const std::string &key, double def=0.0) const;
    int getInt(const std::string &key, int def=0) const;
    std::vector<double> getArray(const std::string &key) const;
    
    // Validation
    void addRule(const ValidationRule &rule);
    std::vector<ValidationError> validate() const;
    bool validateFile(const std::string &configType) const;
    
    // simple map access
    std::map<std::string,std::string> kv;
    std::map<std::string,std::vector<double>> arrays;

private:
    std::vector<ValidationRule> rules;
    static std::map<std::string, std::vector<ValidationRule>> defaultRules;
};

}
