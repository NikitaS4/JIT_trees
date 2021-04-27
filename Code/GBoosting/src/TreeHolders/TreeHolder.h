#ifndef TREE_HOLDER_INCLUDED
#define TREE_HOLDER_INCLUDED

#include "../common/Structs.h"
#include <cstddef>
#include <functional>
#include <string>


class TreeHolder {
public:    
    virtual ~TreeHolder();

    virtual void newTree(const size_t* features, const FVal_t* thresholds,
        const Lab_t* leaves) = 0;
    virtual void popTree() = 0;
    size_t getTreeCount() const;

    virtual Lab_t predictTree(const pytensor1& sample, const size_t treeNum) const = 0;
    virtual void predictTreeFit(const pytensor2& xTrain, const pytensor2& xValid,
        const size_t treeNum, pytensorY& residuals, pytensorY& preds,
        pytensorY& validRes, pytensorY& validPreds) const = 0;
    virtual Lab_t predictAllTrees(const pytensor1& sample) const = 0;
    virtual pytensorY predictAllTrees2d(const pytensor2& sample) const = 0;
    virtual Lab_t predictFromTo(const pytensor1& sample, const size_t from,
        const size_t to) const = 0;

    virtual pytensorY predictTree2d(const pytensor2& xPred, const size_t treeNum) const = 0;
    virtual std::string serialize(const char delimeter, const Lab_t zeroPredictor) const = 0;

    // create needed holder
    static TreeHolder* createHolder(const size_t treeDepth,
        const size_t featureCnt, const size_t threadCnt);
protected:
    // fields
    const size_t treeDepth;
    const size_t innerNodes;
    const size_t featureCnt;
    const size_t leafCnt;
    const size_t threadCnt;
    size_t treeCnt;

    // constructor
    // inheritors need to call ctor
    TreeHolder(const size_t treeDepth, const size_t featureCnt,
        const size_t threadCnt);
};

#endif // TREE_HOLDER_INCLUDED