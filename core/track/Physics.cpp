#include "Physics.hpp"
#include <algorithm>
using namespace pv;

void Physics::cushionReflect(cv::Point2f &pos, cv::Point2f &vel, const cv::Size &tableSize, float radius, float damping){
    if(pos.x - radius < 0 && vel.x < 0){ vel.x = -vel.x * damping; pos.x = radius; }
    if(pos.x + radius > tableSize.width && vel.x > 0){ vel.x = -vel.x * damping; pos.x = tableSize.width - radius; }
    if(pos.y - radius < 0 && vel.y < 0){ vel.y = -vel.y * damping; pos.y = radius; }
    if(pos.y + radius > tableSize.height && vel.y > 0){ vel.y = -vel.y * damping; pos.y = tableSize.height - radius; }
}

std::vector<Physics::Collision> Physics::detectCollisions(const std::vector<Track> &tracks){
    std::vector<Collision> collisions;
    std::mutex collisionMutex;
    
    // Process collision detection in parallel
    cv::parallel_for_(cv::Range(0, (int)tracks.size()), [&](const cv::Range& range) {
        std::vector<Collision> localCollisions;
        
        for(int i = range.start; i < range.end; i++){
            for(size_t j = i + 1; j < tracks.size(); j++){
                const auto &a = tracks[i];
                const auto &b = tracks[j];
                
                // Vector from a to b
                cv::Point2f d = b.c - a.c;
                float dist = cv::norm(d);
                float minDist = a.r + b.r;
                
                if(dist < minDist){
                    Collision c;
                    c.id1 = a.id;
                    c.id2 = b.id;
                    c.normal = d * (1.0f/std::max(dist, 0.001f)); // Normalized direction
                    c.overlap = minDist - dist;
                    localCollisions.push_back(c);
                }
            }
        }
        
        // Merge local results
        if(!localCollisions.empty()){
            std::lock_guard<std::mutex> lock(collisionMutex);
            collisions.insert(collisions.end(), localCollisions.begin(), localCollisions.end());
        }
    });
    
    return collisions;
}

void Physics::resolveCollision(Track &a, Track &b, const cv::Point2f &normal){
    // Relative velocity
    cv::Point2f rv = b.v - a.v;
    
    // Relative velocity along normal
    float velAlongNormal = rv.dot(normal);
    
    // Do not resolve if objects are separating
    if(velAlongNormal > 0) return;
    
    // Calculate impulse scalar
    float j = -(1.0f + RESTITUTION) * velAlongNormal;
    j /= 1.0f/BALL_MASS + 1.0f/BALL_MASS;
    
    // Apply impulse
    cv::Point2f impulse = j * normal;
    a.v -= impulse * (1.0f/BALL_MASS);
    b.v += impulse * (1.0f/BALL_MASS);
}

void Physics::update(std::vector<Track> &tracks, float dt){
    // Detect and resolve collisions
    auto collisions = detectCollisions(tracks);
    
    // Process collisions in parallel chunks
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for(int i = 0; i < (int)collisions.size(); i++){
            const auto &c = collisions[i];
            // Find tracks by ID
            auto it1 = std::find_if(tracks.begin(), tracks.end(), 
                                [&](const Track &t){ return t.id == c.id1; });
            auto it2 = std::find_if(tracks.begin(), tracks.end(), 
                                [&](const Track &t){ return t.id == c.id2; });
            
            if(it1 != tracks.end() && it2 != tracks.end()){
                // Create local copies for thread safety
                Track local1 = *it1;
                Track local2 = *it2;
                
                // Positional correction (prevent sinking)
                float percent = 0.8f;
                float slop = 0.01f;
                cv::Point2f correction = std::max(c.overlap - slop, 0.0f) * percent * 
                                    c.normal / (2.0f);
                local1.c -= correction;
                local2.c += correction;
                
                // Resolve collision velocities
                resolveCollision(local1, local2, c.normal);
                
                // Update original tracks atomically
                #pragma omp critical
                {
                    *it1 = local1;
                    *it2 = local2;
                }
            }
        }
    }
    
    // Update positions and apply friction in parallel
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < (int)tracks.size(); i++){
        auto &t = tracks[i];
        // Update position
        t.c += t.v * dt;
        
        // Apply rolling friction
        float speed = cv::norm(t.v);
        if(speed > 0){
            float frictionMag = FRICTION * BALL_MASS * 9.81f; // F = μmg
            float decel = frictionMag / BALL_MASS * dt; // a = F/m
            
            if(speed <= decel){
                t.v = cv::Point2f(0,0);
            } else {
                t.v *= (1.0f - decel/speed);
            }
        }
    }
}
