#include "JITedTree.h"

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <dlfcn.h>
#include <stdlib.h>


// init static members
const std::string JITedTree::dirPreffix = "JIT_files_model_";
size_t JITedTree::lastModel = 0;


JITedTree::JITedTree(const size_t treeDepth, const size_t featureCnt): 
    treeDepth(treeDepth), innerNodes((1 << treeDepth) - 1), featureCnt(featureCnt),
    leafCnt(size_t(1) << treeDepth), dirName(dirPreffix + std::to_string(lastModel++)), treeCnt(0) {
    // create directory where sources and JIT compiled libs will be stored
    std::string makeDirCmd = "mkdir " + dirName;
    if (system(makeDirCmd.c_str()) != 0)
        throw std::runtime_error("JIT error: can't create directory");
}


void JITedTree::compileTree(const size_t* features, const FVal_t* thresholds,
    const Lab_t* leaves) {
    std::string fname = dirName + "/" + std::to_string(treeCnt);
    ++treeCnt;
    // create file with source code
    createSrc(fname, features, thresholds, leaves);
    // compile to dynamically linked library
    if (compile(fname) != 0)
        throw std::runtime_error("JIT error: compile fault");
    // open compiled library
    openLib(fname);
}


JITedTree::~JITedTree() {
    // close libs
    for (auto & curLib : libPtr) {
        if (curLib)
            dlclose(curLib);
    }
    // delete created sources & compiled libs
    std::string rmDirCmd = "rm -r " + dirName;
    int rc = system(rmDirCmd.c_str()); // rc unused - can't throw from destructor
}


Lab_t JITedTree::predictTree(const pytensor1& sample, const size_t treeNum) const {
    return treePredict[treeNum](sample.data());
}


Lab_t JITedTree::predictAllTrees(const pytensor1& sample) const {
    Lab_t cumSum = 0;
    const double* pData = sample.data();
    for (auto& curPredictor : treePredict)
        cumSum += curPredictor(pData);
    return cumSum;
}


Lab_t JITedTree::predictFromTo(const pytensor1& sample, const size_t from,
    const size_t to) const {
    Lab_t cumSum = 0;
    const double* pData = sample.data();
    for (size_t i = from; i < to; ++i)
        cumSum += treePredict[i](pData);
    return cumSum;
}


size_t JITedTree::getTreeCount() const {
    return treeCnt;
}


void JITedTree::popTree() {
    if (treeCnt > 0) {        
        dlclose(libPtr[treeCnt - 1]);
        treePredict.pop_back();
        libPtr.pop_back();
        --treeCnt;
    }
}


void JITedTree::createSrc(const std::string& fname, const size_t* features, 
    const FVal_t* thresholds, const Lab_t* leaves) {
    std::ofstream jitSrc(fname + ".cpp");
    jitSrc << "#include \"../../src/common/AtomicTypes.h\"\n"
        "#include <cstddef>\n"
        "extern \"C\" Lab_t predict(const double* sample) {\n"
        "static size_t features[] = {";
    for (size_t i = 0; i < treeDepth - 1; ++i) {
        jitSrc << features[i] << ", ";
    }
    jitSrc << features[treeDepth - 1] << "};"
        "static FVal_t thresholds[] = {";
    for (size_t i = 0; i < innerNodes - 1; ++i) {
        jitSrc << thresholds[i] << ", ";
    }
    jitSrc << thresholds[innerNodes - 1] << "};"
        "static Lab_t leaves[] = {";
    for (size_t i = 0; i < leafCnt - 1; ++i) {
        jitSrc << leaves[i] << ", ";
    }
    jitSrc << leaves[leafCnt - 1] << "};\n"
        "size_t curNode = 0;\n"
        "for (size_t h = 0; h < " << treeDepth << "; ++h) {\n"
        "if (sample[features[h]] < thresholds[curNode])\n"
        "   curNode = 2 * curNode + 1;\n"
        "else\n"
        "   curNode = 2 * curNode + 2;\n"
        "}\n"
        "return leaves[curNode - " << innerNodes << "];}\n";
    jitSrc.close();
}


int JITedTree::compile(const std::string& fname) {
    std::string command = "g++ -shared -Wall " + fname + ".cpp -o " + fname + ".so";
    return system(command.c_str());
}


void JITedTree::openLib(const std::string& fname) {
    // open lib
    std::string dlFname = "./" + fname + ".so";
    void* pLib = dlopen(dlFname.c_str(), RTLD_LAZY);
    if (pLib == nullptr)
        throw std::runtime_error("JIT error: can't open .so");
    libPtr.push_back(pLib); // close lib on finish
    // load function
    Lab_t (*predJIT)(const double*) = 
        reinterpret_cast<Lab_t(*)(const double*)>(dlsym(pLib, "predict"));
    if (predJIT == nullptr)
        throw std::runtime_error("JIT error: can't find proper function");
    treePredict.push_back(predJIT);
}
