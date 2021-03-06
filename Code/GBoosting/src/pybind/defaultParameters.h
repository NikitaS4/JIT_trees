#include "../common/Structs.h"


namespace defaultParams {
    const size_t binsMin = 8;
    const size_t binsMax = 512;
    const size_t patience = 3;
    const size_t treeCount = 1000;
    const size_t treeDepth = 7;
    const float learningRate = 0.6f;
    const Lab_t earlyStoppingDelta = 1e-7;
    const float batchPart = 1.0f;
    const unsigned int randomState = 12;
    const bool randomBatches = false;
    const Lab_t regParam = 0; // no regularization
    const bool noEs = false; // use early stopping by default
    const float featureFoldSize = 1.0f; // use all features
    const bool randThresholds = true;
    const bool removeReg = false;
    const size_t threadCnt = 1;
    const bool spoilScores = true;
};
