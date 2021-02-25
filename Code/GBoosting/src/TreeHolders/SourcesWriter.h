#ifndef SOURCES_WRITER_H_INCLUDED
#define SOURCES_WRITER_H_INCLUDED

#include <string>
#include <memory>
#include "../common/Structs.h"
#include "SWTypes.h"


class SourcesWriter {
public:
    virtual ~SourcesWriter();

    virtual void createFile(const std::string& fname,
        const size_t* features, const FVal_t* thresholds,
        const Lab_t* leaves) = 0;

    static std::shared_ptr<SourcesWriter> getInst(const SW_t writerType,
        const size_t treeDepth, const size_t innerNodes,
        const size_t featureCnt, const size_t leafCnt);
protected:
    // fields
    const size_t treeDepth;
    const size_t innerNodes;
    const size_t featureCnt;
    const size_t leafCnt;

    // constants
    static const std::string dexport;

    // methods
    // proteced ctor
    SourcesWriter(const size_t treeDepth, const size_t innerNodes,
        const size_t featureCnt, const size_t leafCnt);
};


#endif  // SOURCES_WRITER_H_INCLUDED
