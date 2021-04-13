#ifndef REGULAR_TREE_INCLUDED
#define REGULAR_TREE_INCLUDED

#include "TreeHolder.h"
#include <vector>


class RegularTree: public TreeHolder {
public:
    RegularTree(const size_t treeDepth, const size_t featureCnt);
    virtual ~RegularTree();

    virtual void newTree(const size_t* features, const FVal_t* thresholds,
        const Lab_t* leaves) final override;
    virtual void popTree() final override;

    virtual Lab_t predictTree(const pytensor1& sample, const size_t treeNum) const final override;
    virtual Lab_t predictAllTrees(const pytensor1& sample) const final override;
    virtual Lab_t predictFromTo(const pytensor1& sample, const size_t from, 
        const size_t to) const final override;

    virtual pytensorY predictTree2d(const pytensor2& xPred, const size_t treeNum) const final override;

private:
    // fields
    std::vector<size_t*> features;
    std::vector<FVal_t*> thresholds;
    std::vector<Lab_t*> leaves;
    size_t threadCnt; // TODO: add setter

    // methods
    inline Lab_t predictTreeRaw(const FVal_t* sample, const size_t treeNum) const;
    inline void validateFeatures();

    inline pytensorY predictTree2dMutlithreaded(const pytensor2& xPred,
        const size_t treeNum) const;

    inline pytensorY predictTree2dSingleThread(const pytensor2& xPred,
        const size_t treeNum) const;

    // constants
    static const int busyWaitMs = 1;
    static constexpr size_t defaultThreadCnt = 1;
};

#endif // REGULAR_TREE_INCLUDED
