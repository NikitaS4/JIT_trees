#include "ParseHelper.h"
#include <cstdlib>


size_t ParseHelper::parseSizeT(const char* start) {
    return std::strtol(start, nullptr, base);
}


double ParseHelper::parseFloat(const char* start) {
    return std::strtod(start, nullptr);
}
