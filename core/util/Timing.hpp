#pragma once
#include <chrono>

namespace pv {

using Clock = std::chrono::high_resolution_clock;

struct Stopwatch {
    Clock::time_point t0 = Clock::now();
    void reset(){ t0 = Clock::now(); }
    double elapsed(){ return std::chrono::duration<double>(Clock::now()-t0).count(); }
};

}
