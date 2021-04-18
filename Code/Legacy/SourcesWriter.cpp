#include "SourcesWriter.h"
#include "SWCycles.h"
#include "SWIfStack.h"


// OS-dependent initializations
#ifdef __linux__
    // linux
    const std::string SourcesWriter::dexport = "";  // empty
#elif _WIN32
    // windows
    const std::string SourcesWriter::dexport = "__declspec(dllexport)";
#else
    // unsupported OS
    int a[-1]; // can not compile for unsupported OS
#endif


SourcesWriter::SourcesWriter(const size_t treeDepth, const size_t innerNodes,
    const size_t featureCnt, const size_t leafCnt): treeDepth(treeDepth),
    innerNodes(innerNodes), featureCnt(featureCnt),
    leafCnt(leafCnt) {}


SourcesWriter::~SourcesWriter() {}


std::shared_ptr<SourcesWriter> SourcesWriter::getInst(const SW_t writerType,
    const size_t treeDepth, const size_t innerNodes,
    const size_t featureCnt, const size_t leafCnt) {
    switch (writerType) {
    case SW_t::BASIC_FOR:
        return std::make_shared<SWCycles>(treeDepth,
            innerNodes, featureCnt, leafCnt);
        break;
    case SW_t::IF_STACK:
        return std::make_shared<SWIfStack>(treeDepth,
            innerNodes, featureCnt, leafCnt);
        break;
    case SW_t::SW_COUNT:
        throw std::runtime_error("SourcesWriter::getInst: wrong writerType");
        break;
    default:
        throw std::runtime_error("SourcesWriter::getInst: wrong writerType");
        break;
    }
    return nullptr;
}
