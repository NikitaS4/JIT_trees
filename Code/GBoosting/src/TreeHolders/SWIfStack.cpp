#include "SWIfStack.h"


SWIfStack::SWIfStack(const size_t treeDepth, const size_t innerNodes,
        const size_t featureCnt, const size_t leafCnt):
        SourcesWriter(treeDepth, innerNodes, featureCnt,
        leafCnt) {}


SWIfStack::~SWIfStack() {}


void SWIfStack::createFile(const std::string& fname,
        const size_t* features, const FVal_t* thresholds,
        const Lab_t* leaves) {
    std::ofstream jitSrc(fname + ".cpp");
    jitSrc <<
        //"#include \"..\\..\\src\\common\\AtomicTypes.h\"\n"
        //"#include <cstddef>\n"
        "#ifdef __cplusplus\n"
        "extern \"C\" \n"
        "#endif\n"
        "double " << dexport << " predict(const double* sample) {\n";
    const size_t currentNode = 0; // the root node
    const size_t currentH = 0; // the root node
    // print tree recursively as if-else operators composition
    addIf(features, thresholds, leaves, currentNode, currentH, jitSrc);
    jitSrc.close();
}


void SWIfStack::addIf(const size_t* features, const FVal_t* thresholds,
        const Lab_t* leaves, const size_t currentNode,
        const size_t currentH, std::ofstream& jitSrc) {
    // print tree recursively
    if (currentH >= treeDepth) {
        // it's a leaf, print return <leaf>
        jitSrc << " return " << leaves[currentNode - innerNodes] << ";\n";
        return;
    }
    // it's not a leaf, print if () { <left branch> } else { <right branch> }
    jitSrc << "if (sample[" << features[currentH] << "]  < " << thresholds[currentNode] << ") {\n";
    // left branch
    addIf(features, thresholds, leaves, 2 * currentNode + 1, 
        currentH + 1, jitSrc);
    jitSrc << "\n} else {\n";
    // right branch
    addIf(features, thresholds, leaves, 2 * currentNode + 2,
        currentH + 1, jitSrc);
    jitSrc << "\n}\n";
}
