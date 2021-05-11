#ifndef TREE_HOLDER_INCLUDED
#define TREE_HOLDER_INCLUDED

#include "../common/Structs.h"
#include <cstddef>
#include <functional>
#include <string>
#include <vector>


class TreeHolder {
public:
    TreeHolder(const size_t treeDepth, const size_t featureCnt,
        const size_t threadCnt);
    virtual ~TreeHolder();

    void newTree(const size_t* features, const FVal_t* thresholds,
        const Lab_t* leaves);
    void popTree();
    size_t getTreeCount() const;

    Lab_t predictTree(const pytensor1& sample, const size_t treeNum) const;
    void predictTreeFit(const pytensor2& xTrain, const pytensor2& xValid,
        const size_t treeNum, pytensorY& residuals, pytensorY& preds,
        pytensorY& validRes, pytensorY& validPreds) const;
    Lab_t predictAllTrees(const pytensor1& sample) const;
    pytensorY predictAllTrees2d(const pytensor2& sample) const;
    Lab_t predictFromTo(const pytensor1& sample, const size_t from,
        const size_t to) const;

    pytensorY predictTree2d(const pytensor2& xPred, const size_t treeNum) const;
    std::string serialize(const char delimeter, const Lab_t zeroPredictor) const;

    // create needed holder
    static TreeHolder* createHolder(const size_t treeDepth,
        const size_t featureCnt, const size_t threadCnt);
    // parse holder from file
    static TreeHolder* parse(const char* repr,
        const std::vector<size_t> delimPos,
        const size_t delimStart, const size_t featureCnt,
        const size_t treeCnt, const size_t treeDepth,
        const size_t threadCnt);
private:
    // fields
    const size_t treeDepth;
    const size_t innerNodes;
    const size_t featureCnt;
    const size_t leafCnt;
    const size_t threadCnt;
    size_t treeCnt;

    std::vector<size_t*> features;
    std::vector<FVal_t*> thresholds;
    std::vector<Lab_t*> leaves;

    // methods
    inline void validateFeatures();
    inline void validateTreeNum(const size_t treeNum) const;

    pytensorY predictTree2dMutlithreaded(const pytensor2& xPred,
        const size_t treeNum) const;

    pytensorY predictTree2dSingleThread(const pytensor2& xPred,
        const size_t treeNum) const;

    inline void predictFitMultithreaded(const pytensor2& xTrain, const pytensor2& xValid,
        const size_t treeNum, pytensorY& residuals, pytensorY& preds,
        pytensorY& validRes, pytensorY& validPreds) const;

    pytensorY allTrees2dMultithreaded(const pytensor2& xPred) const;

    pytensorY predict2dProxy(const pytensor2& xPred,
        const bool allTrees, const size_t treeNum) const;

    std::function<void()> getCallback(const size_t bias,
        const size_t batchSize, const size_t treeNum,
        const pytensor2& xPred, size_t& semThreadsFinish,
        pytensorY& answers) const;

    std::function<void()> getCallbackAll(const size_t bias,
        const size_t batchSize, const size_t treeNum,
        const pytensor2& xPred, size_t& semThreadsFinish,
        pytensorY& answers) const;

    // constants
    static const int busyWaitMs = 1;
};

#endif // TREE_HOLDER_INCLUDED