/*#include "GBMultiPredictor.h"
#include <stdexcept>
#include <iostream>


// TODO: refactor or remove


const size_t GBMultiPredictor::busyWaitDtMs = 10;


GBMultiPredictor::GBMultiPredictor(const size_t threadCnt,
        const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const pytensor2& xTrain,
        const pytensor2& xValid,
        pytensorY& residuals,
        pytensorY& preds,
        pytensorY& validRes,
        pytensorY& validPreds): GBPredcitor(threadCnt, zeroPredictor,
        treeHolder, xTrain, xValid, residuals, preds,
        validRes, validPreds), threadsAlive(true) {
    // ctor
    if (threadCnt <= 1)
        throw std::runtime_error("Wrong thread count passed to GBMultiPredictor");

    setSamplesPerThreadFit();
}


GBMultiPredictor::~GBMultiPredictor() {
    // dtor
}


pytensorY GBMultiPredictor::predict2d(const pytensor2& x) {
    std::cout << "pred2d\n";
    std::vector<std::thread> threads;
    validateFeatureCount(x);
    prepareThreadsPredict(x);

    // usual threads with equal batch sizes
    size_t batchBias = 0;
    for (size_t i = 0; i < threadCnt - 1; ++i) {
        batchBias = i * samplesPerThread;
        threads.push_back(std::thread(threadPredFunc(batchBias,
            samplesPerThread)));
        threads[i].detach();
    }
    // the last thread with the remaining batch part
    batchBias = (threadCnt - 1) * samplesLastThread;
    threads.push_back(std::thread(threadPredFunc(batchBias,
        samplesLastThread)));
    threads[threadCnt - 1].detach();

    // lock myself and wait for all predictions
    waitForThreads();

    return predictAnswer;
}


void GBMultiPredictor::predictTreeTrain(const size_t treeNum) {
    std::cout << "<";
    std::vector<std::thread> threads;
    prepareThreadsFit(treeNum);
    semThreads2 = threadCnt;

    // usual threads with equal batch sizes
    size_t batchBiasTrain = 0;
    size_t batchBiasVal = 0;
    for (size_t i = 0; i < threadCnt - 1; ++i) {
        batchBiasTrain = i * samplesThreadTrain;
        batchBiasVal = i * samplesThreadVal;
        threads.push_back(std::thread(threadFitFunc(batchBiasTrain,
            batchBiasVal, samplesThreadTrain, samplesThreadVal)));
        threads[i].detach();
    }
    // the last thread with the remaining batch part
    batchBiasTrain = (threadCnt - 1) * samplesThreadTrain;
    batchBiasVal = (threadCnt - 1) * samplesThreadVal;
    threads.push_back(std::thread(threadFitFunc(batchBiasTrain,
        batchBiasVal, samplesThreadTrainLast, samplesThreadValLast)));
    threads[threadCnt - 1].detach();

    // lock myself and wait for all predictions
    std::cout << "before waiting\n";
    waitForThreads();
    std::cout << ">\n";
}


std::function<void()> GBMultiPredictor::threadFitFunc(const size_t threadBiasTrain,
    const size_t threadBiasVal, const size_t threadTrainCnt,
    const size_t threadValCnt) {
    auto threadFunc = [&, threadBiasTrain, threadBiasVal,
        threadTrainCnt, threadValCnt]() mutable {
        try {
        std::cout << "thread open\n";
        Lab_t prediction = 0;
        // cycle for x_train
        const size_t trainLim = threadBiasTrain + threadTrainCnt;
        for (size_t i = threadBiasTrain; i < trainLim; ++i) {
            prediction = treeHolder.predictTree(xt::row(xTrain, i), 
			            curTreeNum);
		    residuals(i) -= prediction;
		    preds(i) += prediction;
        }
        std::cout << "preds for train got\n";
        semThreads2 -= 1;
        while (semThreads2 > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        // cycle for x_valid
        const size_t valLim = threadBiasVal + threadValCnt;
        for (size_t j = threadBiasVal; j < valLim; ++j) {
            prediction = treeHolder.predictTree(xt::row(xValid, j), 
			            curTreeNum);
		    validRes(j) -= prediction;
		    validPreds(j) += prediction;
        }
        std::cout << "preds for valid got\n";

        semThreadsFinished -= 1; // release semaphore for me
        }
        catch (std::runtime_error& err) {
            std::cout << "an exception thrown: " << err.what() << "\n";
        }
        semThreadsFinished -= 1;
        std::cout << "thread close\n";
    };
    return threadFunc;
}


std::function<void()> GBMultiPredictor::threadPredFunc(const size_t threadBias,
    const size_t batchSize) {
    auto threadFunc = [&, threadBias, batchSize]() mutable {
        const size_t upperLim = threadBias + batchSize;
        for (size_t i = threadBias; i < upperLim; ++i) {
            predictAnswer(i) = zeroPredictor + 
                treeHolder.predictAllTrees(xt::row(*xToPredict, i));
        }

        semThreadsFinished -= 1; // release semaphore for me
    };
    return threadFunc;
}


void GBMultiPredictor::prepareThreadsFit(const size_t treeNum) {
    curTreeNum = treeNum;
    semThreadsFinished = threadCnt; // each thread has to decrement
}


void GBMultiPredictor::waitForThreads() {
    // lock main thread until each tread completes work
    while (semThreadsFinished > 0) {
        // busy wait
        std::this_thread::sleep_for(std::chrono::milliseconds(busyWaitDtMs));
    }
}


void GBMultiPredictor::prepareThreadsPredict(const pytensor2& x) {    
    // prepare variable for the answer
    size_t sampleCnt = x.shape(0);
    predictAnswer = xt::zeros<Lab_t>({sampleCnt});
    // share x between the threads
    xToPredict = &x;
    setSamplesPerThread(sampleCnt);
    semThreadsFinished = threadCnt; // each thread has to decrement
}


void GBMultiPredictor::setSamplesPerThread(const size_t sampleCnt) {
    samplesPerThread = sampleCnt / threadCnt;
    samplesLastThread = sampleCnt - (samplesPerThread * threadCnt);
}


void GBMultiPredictor::setSamplesPerThreadFit() {
    samplesThreadTrain = trainLen / threadCnt;
    samplesThreadTrainLast = trainLen - (samplesThreadTrain * (threadCnt - 1));

    samplesThreadVal = validLen / threadCnt;
    samplesThreadValLast = validLen - (samplesThreadVal * (threadCnt - 1));
}
*/
