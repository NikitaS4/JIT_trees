#ifndef SW_IF_STACK_H_INCLUDED
#define SW_IF_STACK_H_INCLUDED

#include "SourcesWriter.h"
#include <iostream>
#include <fstream>


// Tree is traversed with FOR cycle (height changes from root to leaf)
class SWIfStack: public SourcesWriter {
public:
    SWIfStack(const size_t treeDepth, const size_t innerNodes,
        const size_t featureCnt, const size_t leafCnt);

    virtual ~SWIfStack();

    virtual void createFile(const std::string& fname,
        const size_t* features, const FVal_t* thresholds,
        const Lab_t* leaves) override;
private:
    void addIf(const size_t* features, const FVal_t* thresholds,
        const Lab_t* leaves, const size_t currentNode, 
        const size_t currentH, std::ofstream& jitSrc);
};


#endif  // SW_IF_STACK_H_INCLUDED