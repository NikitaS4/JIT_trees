#include "RegularTree.h"
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>


RegularTree::RegularTree(const size_t treeDepth, const size_t featureCnt):
    TreeHolder(treeDepth, featureCnt), threadCnt(defaultThreadCnt) {
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

    // this will fix errors
    // TODO: find out the reason for the wrong values
    validateFeatures();
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
    //const FVal_t* samplePtr = sample.data(); // get data in array-pointer format
    //return predictTreeRaw(samplePtr, treeNum);

    validateTreeNum(treeNum);
    // get pointers for faster access
    const size_t* curFeatures = features[treeNum];
    const FVal_t* curThresholds = thresholds[treeNum];

    size_t curNode = 0;
    // tree traverse
    for (size_t h = 0; h < treeDepth; ++h) {
        if (sample(curFeatures[h]) < curThresholds[curNode])
            curNode = 2 * curNode + 1;
        else
            curNode = 2 * curNode + 2;
    }
    return leaves[treeNum][curNode - innerNodes];
}


Lab_t RegularTree::predictAllTrees(const pytensor1& sample) const {
    Lab_t curSum = 0;
    //const FVal_t* samplePtr = sample.data(); // get data in array-pointer format
    for (size_t i = 0; i < treeCnt; ++i)
        curSum += predictTree(sample, i);
    return curSum;
}


Lab_t RegularTree::predictFromTo(const pytensor1& sample, const size_t from,
    const size_t to) const {
    Lab_t curSum = 0;
    //const FVal_t* samplePtr = sample.data(); // get data in array-pointer format
    for (size_t i = from; i < to; ++i)
        curSum += predictTree(sample, i);
    return curSum;
}


pytensorY RegularTree::predictTree2d(const pytensor2& xPred,
    const size_t treeNum) const {
    validateTreeNum(treeNum);
    if (threadCnt > 1)
        return predictTree2dMutlithreaded(xPred, treeNum);
    else
        return predictTree2dSingleThread(xPred, treeNum);
}


void RegularTree::validateFeatures() {
    for (auto & curFeatureArr: features) {
        for (size_t h = 0; h < treeDepth; ++h) {
            if (curFeatureArr[h] >= featureCnt)
                // too big value
                curFeatureArr[h] = featureCnt - 1;
        }
    }
}


void RegularTree::validateTreeNum(const size_t treeNum) const {
    if (treeNum >= features.size())
        throw std::runtime_error("wrong treeNum");
    if (treeNum >= thresholds.size())
        throw std::runtime_error("wrong treeNum");
    if (treeNum >= leaves.size())
        throw std::runtime_error("wrong treeNum");
}


pytensorY RegularTree::predictTree2dSingleThread(const pytensor2& xPred,
    const size_t treeNum) const {
    // get pointers for faster access
    const size_t* curFeatures = features[treeNum];
    const FVal_t* curThresholds = thresholds[treeNum];
    const size_t sampleCnt = xPred.shape(0);

    // tensor to store and return predictions
    pytensorY answers = xt::zeros<Lab_t>({sampleCnt});

    size_t curNode = 0;
    for (size_t i = 0; i < sampleCnt; ++i) {
        // decision tree traverse
        for (size_t h = 0; h < treeDepth; ++h) {
            if (xPred(i, curFeatures[h]) < curThresholds[curNode])
                curNode = 2 * curNode + 1;
            else
                curNode = 2 * curNode + 2;
        }
        answers(i) = leaves[treeNum][curNode - innerNodes];
        // remember to set curNode to 0 before the next step
        curNode = 0;
    }
    return answers;
}


pytensorY RegularTree::predictTree2dMutlithreaded(const pytensor2& xPred,
    const size_t treeNum) const {

    std::vector<std::thread> threads;
    // init "semaphore" with the max value
    // each thread will decrement the semaphore before finish
    size_t semThreadsFinish = threadCnt;
    static const size_t semUnlocked = 0;

    // each thread will predict on it's own batch
    size_t bias; // bias of the batch
    size_t batchSize = xPred.shape(0) / threadCnt;
    // the size of the last batch (it may differ)
    size_t lastBatchSize = xPred.shape(0) - (threadCnt - 1) * batchSize;
    // tensor to store and return predictions
    pytensorY answers = xt::zeros<Lab_t>({xPred.shape(0)});
    // get pointers for faster access
    const size_t* curFeatures = features[treeNum];
    const FVal_t* curThresholds = thresholds[treeNum];
    // create threads for prediction
    for (size_t i = 0; i < threadCnt - 1; ++i) {
        bias = i * batchSize; // compute batch bias
        // define thread callback
        // each thread has it's own unique callback
        // (to avoid data races)
        auto threadCallback = [&, bias, batchSize]() mutable {
            // compute the end of the batch
            const size_t upperLimit = batchSize + bias;
            size_t curNode = 0; // current node in the decision tree
            // predict for all batch members
            for (size_t j = bias; j < upperLimit; ++j) {
                // decision tree traverse
                for (size_t h = 0; h < treeDepth; ++h) {
                    if (xPred(j, curFeatures[h]) < curThresholds[curNode])
                        curNode = 2 * curNode + 1;
                    else
                        curNode = 2 * curNode + 2;
                }
                answers(j) = leaves[treeNum][curNode - innerNodes];
                // remember to set curNode to 0 before the next step
                curNode = 0;
            }
            // thread is finishing, release "semaphore"
            semThreadsFinish -= 1;
        };
        // create a thread
        threads.push_back(std::thread(threadCallback));
        threads[i].detach(); // launch the thread
    }
    // prepare the last thread (it can have different batchSize)
    bias = (threadCnt - 1) * batchSize;
    threads.push_back(std::thread([&, bias, batchSize]() mutable {
        const size_t upperLimit = lastBatchSize + bias;
        size_t curNode = 0;
        for (size_t j = bias; j < upperLimit; ++j) {
            for (size_t h = 0; h < treeDepth; ++h) {
                if (xPred(j, curFeatures[h]) < curThresholds[curNode])
                    curNode = 2 * curNode + 1;
                else
                    curNode = 2 * curNode + 2;
            }
            answers(j) = leaves[treeNum][curNode - innerNodes];
            curNode = 0;
        }
        semThreadsFinish -= 1;
    }));
    threads[threadCnt - 1].detach();

    while (semThreadsFinish > semUnlocked) {
        // busy wait (until threads finishes predictions)
        std::this_thread::sleep_for(std::chrono::milliseconds(busyWaitMs));
    }
    // now all predictions have been aggregated, can return an answer

    return answers;
}