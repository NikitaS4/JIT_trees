#ifndef HISTORYADAPTER_H
#define HISTORYADAPTER_H

#include "Structs.h"
#include "../common/History.h"


namespace py = pybind11;


namespace Adapter {
    class History {
    public:
    History(::History&& history);

    // getters only
    size_t getTreesLearnt();
    pyarrayY getTrainLosses();
    pyarrayY getValidLosses();
    private:
    ::History history;
    };
};

#endif
