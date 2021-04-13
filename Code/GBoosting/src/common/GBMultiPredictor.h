// TODO: refactor or remove
/*#ifndef GBMULTIPREDICTOR_H_INCLUDED
#define GBMULTIPREDICTOR_H_INCLUDED


#include "GBPredictor.h"
#include <thread>
#include <vector>
#include <functional>


class GBMultiPredictor: public GBPredcitor {
public:
    GBMultiPredictor(const size_t threadCnt,
        const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const pytensor2& xTrain,
        const pytensor2& xValid,
        pytensorY& residuals,
        pytensorY& preds,
        pytensorY& validRes,
        pytensorY& validPreds);
    virtual ~GBMultiPredictor();

    pytensorY predict2d(const pytensor2& x) override;
    
    void predictTreeTrain(const size_t treeNum) override;

private:
    size_t semThreadsFinished; // semaphore, 0 if all threads completed their tasks
    size_t semThreads2;
    pytensor2 const * xToPredict;
    pytensorY predictAnswer;
    bool threadsAlive;
    int threadsAction;
    size_t samplesPerThread;
    size_t samplesLastThread;
    
    size_t samplesThreadTrain;
    size_t samplesThreadTrainLast;

    size_t samplesThreadVal;
    size_t samplesThreadValLast;

    size_t curTreeNum;

    static const size_t busyWaitDtMs;

    std::function<void()> threadFitFunc(const size_t threadBiasTrain,
        const size_t threadBiasVal, const size_t threadTrainCnt,
        const size_t threadValCnt);
    std::function<void()> threadPredFunc(const size_t threadBias,
        const size_t batchSize);
    inline void prepareThreadsFit(const size_t treeNum);
    void waitForThreads();
    inline void prepareThreadsPredict(const pytensor2& x);
    inline void setSamplesPerThread(const size_t sampleCnt);
    inline void setSamplesPerThreadFit();
};


#endif // GBMULTIPREDICTOR_H_INCLUDED
*/
