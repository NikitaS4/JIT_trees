#ifndef REGULAR_TREE_INCLUDED
#define REGULAR_TREE_INCLUDED

#include "TreeHolder.h"
#include <vector>
#include <functional>


class RegularTree: public TreeHolder {
public:
    RegularTree(const size_t treeDepth, const size_t featureCnt);
    virtual ~RegularTree();

    virtual void newTree(const size_t* features, const FVal_t* thresholds,
        const Lab_t* leaves) final override;
    virtual void popTree() final override;

    virtual Lab_t predictTree(const pytensor1& sample, const size_t treeNum) const final override;
    virtual void predictTreeFit(const pytensor2& xTrain, const pytensor2& xValid,
        const size_t treeNum, pytensorY& residuals, pytensorY& preds,
        pytensorY& validRes, pytensorY& validPreds) const final override;
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
    inline void validateFeatures();
    inline void validateTreeNum(const size_t treeNum) const;

    inline pytensorY predictTree2dMutlithreaded(const pytensor2& xPred,
        const size_t treeNum) const;

    pytensorY predictTree2dSingleThread(const pytensor2& xPred,
        const size_t treeNum) const;

    inline void predictFitMultithreaded(const pytensor2& xTrain, const pytensor2& xValid,
        const size_t treeNum, pytensorY& residuals, pytensorY& preds,
        pytensorY& validRes, pytensorY& validPreds) const;

    std::function<void()> getCallback(const size_t bias,
        const size_t batchSize, const size_t treeNum,
        const pytensor2& xPred, size_t& semThreadsFinish,
        pytensorY& answers) const;

    // constants
    static const int busyWaitMs = 1;
    static constexpr size_t defaultThreadCnt = 3;
};

#endif // REGULAR_TREE_INCLUDED
