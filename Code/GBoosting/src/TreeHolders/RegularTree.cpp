#include "RegularTree.h"
#include <cstring>
#include <iostream>


RegularTree::RegularTree(const size_t treeDepth, const size_t featureCnt):
    TreeHolder(treeDepth, featureCnt) {
    // ctor
}


RegularTree::~RegularTree() {
    // dtor
    for (size_t i = 0; i < treeCnt; ++i) {
        delete [] features[i];
        delete [] thresholds[i];
        delete [] leaves[i];

        features[i] = nullptr;
        thresholds[i] = nullptr;
        leaves[i] = nullptr;
    }
}


void RegularTree::newTree(const size_t* features, const FVal_t* thresholds,
    const Lab_t* leaves) {
    static const size_t featuresSize = treeDepth * sizeof(*features);
    static const size_t thresholdSize = innerNodes * sizeof(*thresholds);
    static const size_t leavesSize = leafCnt * sizeof(*leaves);
    ++treeCnt;
    // copy arrays
    this->features.push_back((size_t*)(std::memcpy(new size_t[treeDepth], features, featuresSize)));
    this->thresholds.push_back((FVal_t*)(std::memcpy(new FVal_t[innerNodes], thresholds, thresholdSize)));
    this->leaves.push_back((Lab_t*)(std::memcpy(new Lab_t[leafCnt], leaves, leavesSize)));    
}


void RegularTree::popTree() {
    // decrease tree count
    --treeCnt;

    // free memory
    delete [] features[treeCnt];
    delete [] thresholds[treeCnt];
    delete [] leaves[treeCnt];

    // assign nulls to pointers
    features[treeCnt] = nullptr;
    thresholds[treeCnt] = nullptr;
    leaves[treeCnt] = nullptr;

    // pop vectors
    features.pop_back();
    thresholds.pop_back();
    leaves.pop_back();
}


Lab_t RegularTree::predictTree(const pytensor1& sample, 
    const size_t treeNum) const {
    const FVal_t* samplePtr = sample.data(); // get data in array-pointer format
    return predictTreeRaw(samplePtr, treeNum);
}


Lab_t RegularTree::predictAllTrees(const pytensor1& sample) const {
    Lab_t curSum = 0;
    const FVal_t* samplePtr = sample.data(); // get data in array-pointer format
    for (size_t i = 0; i < treeCnt; ++i)
        curSum += predictTreeRaw(samplePtr, i);
    return curSum;
}


Lab_t RegularTree::predictFromTo(const pytensor1& sample, const size_t from,
    const size_t to) const {
    Lab_t curSum = 0;
    const FVal_t* samplePtr = sample.data(); // get data in array-pointer format
    for (size_t i = from; i < to; ++i)
        curSum += predictTreeRaw(samplePtr, i);
    return curSum;
}


Lab_t RegularTree::predictTreeRaw(const FVal_t* sample, 
    const size_t treeNum) const {
    // use const double* as arg to call tensor's data() method only once
    // in predictAllTrees or predictFromTo methods
    size_t curNode = 0;
    
    // get pointers for faster access
    const size_t* curFeatures = features[treeNum];
    const FVal_t* curThresholds = thresholds[treeNum];

    for (size_t h = 0; h < treeDepth; ++h) {
        if (sample[curFeatures[h]] < curThresholds[curNode])
            curNode = 2 * curNode + 1;
        else
            curNode = 2 * curNode + 2;
    }    
    return leaves[treeNum][curNode - innerNodes];
}
