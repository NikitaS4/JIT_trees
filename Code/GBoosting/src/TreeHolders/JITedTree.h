#ifndef JITED_TREE_H
#define JITED_TREE_H

#include "TreeHolder.h"
#include <cstddef>
#include <vector>
#include <string>

// OS-dependent imports
#ifdef _WIN32
    // windows
    // import windows.h for HINSTANCE type
    #include <windows.h>
#endif


class JITedTree: public TreeHolder {
public:
    JITedTree(const size_t treeDepth, const size_t featureCnt);

    /*JITedTree(const size_t treeDepth, const size_t innerNodes, 
        const size_t featureCnt, const size_t leafCnt, const size_t* features, 
        const FVal_t* thresholds, const Lab_t* leaves);
    JITedTree();
    JITedTree(JITedTree && other) noexcept;*/
    virtual ~JITedTree();

    virtual void newTree(const size_t* features, const FVal_t* thresholds, 
        const Lab_t* leaves) final override;
    virtual void popTree() final override;

    virtual Lab_t predictTree(const pytensor1& sample, const size_t treeNum) const final override;
    virtual Lab_t predictAllTrees(const pytensor1& sample) const final override;
    virtual Lab_t predictFromTo(const pytensor1& sample, const size_t from, 
        const size_t to) const final override;

private:
    // fields
    const std::string dirName;

    // OS-dependent pointer types
    #ifdef __linux__
        // linux
        std::vector<void*> libPtr;
    #elif _WIN32
        // windows
        std::vector<HINSTANCE> libPtr;
    #else // unsupported OS
        int a[-1]; // can not compile for unsupported OS
    #endif
    std::vector<Lab_t(*)(const double*)> treePredict;
    
    // need count models to create different dirs with JIT files for each model
    static size_t lastModel;

    // constants
    static const std::string dirPreffix;
    static const std::string rmExe;
    static const std::string dexport;
    static const std::string dirDelimeter;

    // methods
    void inline createSrc(const std::string& fname,
        const size_t* features, const FVal_t* thresholds, const Lab_t* leaves);
    int inline compile(const std::string& fname);
    void inline openLib(const std::string& fname);
};

#endif // JITED_TREE_H
