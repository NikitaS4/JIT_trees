#ifndef SW_CYCLES_H_INCLUDED
#define SW_CYCLES_H_INCLUDED

#include "SourcesWriter.h"


// Tree is traversed with FOR cycle (height changes from root to leaf)
class SWCycles: public SourcesWriter {
public:
    SWCycles(const size_t treeDepth, const size_t innerNodes,
        const size_t featureCnt, const size_t leafCnt);

    virtual ~SWCycles();

    virtual void createFile(const std::string& fname,
        const size_t* features, const FVal_t* thresholds,
        const Lab_t* leaves) override;
private:
};


#endif  // SW_CYCLES_H_INCLUDED
