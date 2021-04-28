#ifndef PARSEHELPER_H_INCLUDED
#define PARSEHELPER_H_INCLUDED

#include <cstddef>


class ParseHelper {
public:
    static size_t parseSizeT(const char* start);
    static double parseFloat(const char* start);
private:
    // only static members
    ParseHelper() = delete;
    ~ParseHelper() = delete;

    // consts
    static const int base = 10;
};

#endif // PARSEHELPER_H_INCLUDED
