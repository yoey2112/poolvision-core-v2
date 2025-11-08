#include "Config.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

using namespace pv;

std::map<std::string, std::vector<Config::ValidationRule>> Config::defaultRules = {
    {"table", {
        {"table_width", true, [](const Config &c){ 
            int w = c.getInt("table_width"); 
            return w >= 1000 && w <= 5000; 
        }, "Table width must be between 1000 and 5000"},
        {"table_height", true, [](const Config &c){ 
            int h = c.getInt("table_height"); 
            return h >= 500 && h <= 2500; 
        }, "Table height must be between 500 and 2500"},
        {"ball_radius_px", true, [](const Config &c){ 
            int r = c.getInt("ball_radius_px"); 
            return r >= 5 && r <= 50; 
        }, "Ball radius must be between 5 and 50 pixels"},
        {"homography", true, [](const Config &c){ 
            auto h = c.getArray("homography"); 
            return h.size() == 9; 
        }, "Homography must be a 3x3 matrix (9 values)"},
        {"pockets", true, [](const Config &c){ 
            auto p = c.getArray("pockets"); 
            return !p.empty(); 
        }, "Must define at least one pocket"}
    }},
    {"camera", {
        {"width", true, [](const Config &c){ 
            int w = c.getInt("width"); 
            return w >= 640 && w <= 3840; 
        }, "Camera width must be between 640 and 3840"},
        {"height", true, [](const Config &c){ 
            int h = c.getInt("height"); 
            return h >= 480 && h <= 2160; 
        }, "Camera height must be between 480 and 2160"},
        {"fps", true, [](const Config &c){ 
            int fps = c.getInt("fps"); 
            return fps >= 15 && fps <= 120; 
        }, "FPS must be between 15 and 120"}
    }},
    {"colors", {
        {"prototypes", true, [](const Config &c){ 
            bool hasValidPrototypes = false;
            for(const auto &kv: c.kv){
                auto arr = c.getArray(kv.first);
                if(arr.size() == 3 && 
                   arr[0] >= 0 && arr[0] <= 100 && // L
                   arr[1] >= -128 && arr[1] <= 127 && // a
                   arr[2] >= -128 && arr[2] <= 127) { // b
                    hasValidPrototypes = true;
                }
            }
            return hasValidPrototypes;
        }, "Must define at least one valid LAB color prototype (L: 0-100, a/b: -128-127)"}
    }}
};

static std::string trim(const std::string &s){
    size_t a = s.find_first_not_of(" \t\r\n");
    if(a==std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b-a+1);
}

bool Config::load(const std::string &path){
    std::ifstream in(path);
    if(!in) return false;
    std::string line;
    std::string currentKey;
    while(std::getline(in,line)){
        line = trim(line);
        if(line.empty()) continue;
        if(line[0]=='#') continue;
        auto colon = line.find(":");
        if(colon!=std::string::npos){
            std::string key = trim(line.substr(0,colon));
            std::string val = trim(line.substr(colon+1));
            // array
            if(!val.empty() && val.front()=='['){
                // flatten numbers inside [] and nested lists
                std::vector<double> nums;
                std::string s = val;
                // read until matching ']' possibly multi-line
                while(s.find(']')==std::string::npos && std::getline(in,line)){
                    s += line;
                }
                // remove brackets and non-numeric chars except , - .
                for(char &c: s) if(c=='['||c==']') c=' ';
                std::istringstream iss(s);
                double x;
                while(iss>>x) nums.push_back(x);
                arrays[key] = nums;
            } else {
                // plain value
                // remove quotes
                if(!val.empty() && val.front()=='"' && val.back()=='"') val = val.substr(1,val.size()-2);
                kv[key]=val;
            }
        }
    }
    return true;
}

std::string Config::getString(const std::string &key, const std::string &def) const{
    auto it = kv.find(key);
    if(it==kv.end()) return def;
    return it->second;
}

double Config::getDouble(const std::string &key, double def) const{
    auto it = kv.find(key);
    if(it==kv.end()) return def;
    try{ return std::stod(it->second);}catch(...){return def;}
}

int Config::getInt(const std::string &key, int def) const{
    return (int)getDouble(key, def);
}

std::vector<double> Config::getArray(const std::string &key) const{
    auto it = arrays.find(key);
    if(it==arrays.end()) return {};
    return it->second;
}

void Config::addRule(const ValidationRule &rule) {
    rules.push_back(rule);
}

std::vector<Config::ValidationError> Config::validate() const {
    std::vector<ValidationError> errors;
    
    for (const auto& rule : rules) {
        if (rule.required) {
            // Check if key exists
            bool exists = kv.find(rule.key) != kv.end() || arrays.find(rule.key) != arrays.end();
            if (!exists) {
                errors.push_back({rule.key, "Required key '" + rule.key + "' is missing"});
                continue;
            }
        }
        
        // Run custom validation function if provided
        if (rule.validate && !rule.validate(*this)) {
            errors.push_back({rule.key, rule.description});
        }
    }
    
    return errors;
}

bool Config::validateFile(const std::string &configType) const {
    auto it = defaultRules.find(configType);
    if (it == defaultRules.end()) {
        return true; // No rules defined for this config type
    }
    
    const auto& typeRules = it->second;
    for (const auto& rule : typeRules) {
        if (rule.required) {
            bool exists = kv.find(rule.key) != kv.end() || arrays.find(rule.key) != arrays.end();
            if (!exists) {
                return false;
            }
        }
        
        if (rule.validate && !rule.validate(*this)) {
            return false;
        }
    }
    
    return true;
}
