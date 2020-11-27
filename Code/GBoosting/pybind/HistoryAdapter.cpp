#include "HistoryAdapter.h"
#include "ArrayAdapter.h"
#include <utility>


namespace Adapter {

History::History(::History&& history): history(std::forward<::History>(history)) {}

size_t History::getTreesLearnt() {
    return history.getTreesLearnt();
}

pyarrayY History::getTrainLosses() {        
    return ArrayAdapter::labelsToPy(history.getTrainLosses());
}

pyarrayY History::getValidLosses() {
    return ArrayAdapter::labelsToPy(history.getValidLosses());
}

};