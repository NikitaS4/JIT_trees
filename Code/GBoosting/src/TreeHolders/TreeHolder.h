#ifndef TREE_HOLDER_INCLUDED
#define TREE_HOLDER_INCLUDED

#include "../common/Structs.h"
#include "SWTypes.h"
#include <cstddef>
#include <functional>


class TreeHolder {
public:    
    virtual ~TreeHolder();

    virtual void newTree(const size_t* features, const FVal_t* thresholds,
        const Lab_t* leaves) = 0;
    virtual void popTree() = 0;
    size_t getTreeCount() const;

    virtual Lab_t predictTree(const pytensor1& sample, const size_t treeNum) const = 0;
    virtual Lab_t predictAllTrees(const pytensor1& sample) const = 0;
    virtual Lab_t predictFromTo(const pytensor1& sample, const size_t from,
        const size_t to) const = 0;

    virtual pytensorY predictTree2d(const pytensor2& xPred, const size_t treeNum) const = 0;

    // create needed holder
    static TreeHolder* createHolder(const bool JITed, 
        const size_t treeDepth, const size_t featureCnt,
        const SW_t JITedCodeType);
protected:
    // fields
    const size_t treeDepth;
    const size_t innerNodes;
    const size_t featureCnt;
    const size_t leafCnt;
    size_t treeCnt;

    // constructor
    // inheritors need to call ctor
    TreeHolder(const size_t treeDepth, const size_t featureCnt);
};

#endif // TREE_HOLDER_INCLUDED