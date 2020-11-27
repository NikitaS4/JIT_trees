#ifndef ARRAYADAPTER_H
#define ARRAYADAPTER_H

#include <vector>
#include <stdexcept>

#include "Structs.h"


namespace Adapter {
    class ArrayAdapter {
    public:
        // numpy arrays to std::vectors
        static std::vector<Lab_t> labelsToVector(pyarrayY array);
        static std::vector<FVal_t> featuresToVector(pyarray array);
        static std::vector<std::vector<FVal_t>> featuresToMtx(pyarray array);

        // std::vectors to numpy arrays
        static pyarrayY labelsToPy(const std::vector<Lab_t>& array);
        static pyarray featuresToPy(const std::vector<FVal_t>& array);
        static pyarray featureMtxToPy(const std::vector<std::vector<FVal_t>>& array);
    private:
        // static class - delete constructors
        ArrayAdapter() {};
        ~ArrayAdapter() {};
    };
};

#endif // ARRAYADAPTER_H
