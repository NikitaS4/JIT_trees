#include "SWCycles.h"
#include <iostream>
#include <fstream>


SWCycles::SWCycles(const size_t treeDepth, const size_t innerNodes,
    const size_t featureCnt, const size_t leafCnt):
    SourcesWriter(treeDepth, innerNodes, featureCnt,
    leafCnt) {}


SWCycles::~SWCycles() {}


void SWCycles::createFile(const std::string& fname,
    const size_t* features, const FVal_t* thresholds,
    const Lab_t* leaves) {
    std::ofstream jitSrc(fname + ".cpp");
    jitSrc <<
        //"#include \"..\\..\\src\\common\\AtomicTypes.h\"\n"
        //"#include <cstddef>\n"
        "#ifdef __cplusplus\n"
        "extern \"C\" \n"
        "#endif\n"
        "double " << dexport << " predict(const double* sample) {\n"
        "static unsigned int features[] = {";
    for (size_t i = 0; i < treeDepth - 1; ++i) {
        jitSrc << features[i] << ", ";
    }
    jitSrc << features[treeDepth - 1] << "};"
        "static double thresholds[] = {";
    for (size_t i = 0; i < innerNodes - 1; ++i) {
        jitSrc << thresholds[i] << ", ";
    }
    jitSrc << thresholds[innerNodes - 1] << "};"
        "static double leaves[] = {";
    for (size_t i = 0; i < leafCnt - 1; ++i) {
        jitSrc << leaves[i] << ", ";
    }
    jitSrc << leaves[leafCnt - 1] << "};\n"
        "unsigned int curNode = 0;\n"
        "for (unsigned int h = 0; h < " << treeDepth << "; ++h) {\n"
        "if (sample[features[h]] < thresholds[curNode])\n"
        "   curNode = 2 * curNode + 1;\n"
        "else\n"
        "   curNode = 2 * curNode + 2;\n"
        "}\n"
        "return leaves[curNode - " << innerNodes << "];}\n";
    jitSrc.close();
}