#ifndef JITED_TREE_H
#define JITED_TREE_H

#include "../common/Structs.h"
#include <cstddef>
#include <vector>
#include <string>


class JITedTree {
public:
    JITedTree(const size_t treeDepth, const size_t featureCnt);

    JITedTree(const size_t treeDepth, const size_t innerNodes, 
        const size_t featureCnt, const size_t leafCnt, const size_t* features, 
        const FVal_t* thresholds, const Lab_t* leaves);
    JITedTree();
    JITedTree(JITedTree && other) noexcept;
    virtual ~JITedTree();

    void compileTree(const size_t* features, const FVal_t* thresholds, 
        const Lab_t* leaves);
    void popTree();
    size_t getTreeCount() const;

    Lab_t predictTree(const pytensor1& sample, const size_t treeNum) const;
    Lab_t predictAllTrees(const pytensor1& sample) const;
    Lab_t predictFromTo(const pytensor1& sample, const size_t from, 
        const size_t to) const;

private:
    // fields
    const size_t treeDepth;
    const size_t innerNodes;
    const size_t featureCnt;
    const size_t leafCnt;
    const std::string dirName;
    size_t treeCnt;
    std::vector<void*> libPtr;
    std::vector<Lab_t(*)(const double*)> treePredict;
    
    // need count models to create different dirs with JIT files for each model
    static size_t lastModel;

    // constants
    static const std::string dirPreffix;

    // methods
    void inline createSrc(const std::string& fname,
        const size_t* features, const FVal_t* thresholds, const Lab_t* leaves);
    int inline compile(const std::string& fname);
    void inline openLib(const std::string& fname);
};

#endif // JITED_TREE_H
