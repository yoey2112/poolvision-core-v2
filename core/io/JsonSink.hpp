#pragma once
#include <string>
#include "../util/Types.hpp"

namespace pv {

class JsonSink {
public:
    void emit(const FrameState &s);
};

}
