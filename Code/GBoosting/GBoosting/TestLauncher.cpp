#include "TestLauncher.h"
#include "History.h"

TestLauncher::TestLauncher(const std::vector<std::vector<FVal_t>>& xTrain,
	const std::vector<Lab_t>& yTrain,
	const std::vector<std::vector<FVal_t>>& xValid,
	const std::vector<Lab_t>& yValid) : xTrain(xTrain), yTrain(yTrain),
	xValid(xValid), yValid(yValid) {}

TestLauncher::TestLauncher(const std::string& xTrainFile,
	const std::string& yTrainFile,
	const std::string& xValidFile,
	const std::string& yValidFile) {
	xTrain = DataLoader::loadX(xTrainFile);
	yTrain = DataLoader::loadY(yTrainFile);
	xValid = DataLoader::loadX(xValidFile);
	yValid = DataLoader::loadY(yValidFile);
}

void TestLauncher::performTest(const std::vector<size_t>& treeCounts,
	const std::vector<size_t>& treeDepths,
	const std::vector<size_t>& binCounts,
	const std::vector<float>& learnRates,
	const size_t commonPatience) const {
	std::cout << "Test launched\n";
	std::cout << "Train data: " << xTrain.size() << " samples\n";
	std::cout << "Validation data: " << xValid.size() << " samples\n";
	std::cout << "Feature count: " << xTrain[0].size() << "\n";
	std::cout << "Patience: " << commonPatience << "\n";

	size_t experimentNumber = 1;
	Lab_t bestScore = 0;
	Lab_t curScore = 0;
	bool bestFound = false;
	for (auto& treeNum : treeCounts) {
		for (auto& depth : treeDepths) {
			for (auto& bins : binCounts) {
				for (auto& lr : learnRates) {
					std::cout << "Experiment " << experimentNumber << "\n";
					std::cout << "===================================\n";
					printConditions(treeNum, depth, bins, lr);

					GradientBoosting model(bins, commonPatience);
					std::cout << "Fit\n";
					History history = model.fit(xTrain, yTrain, 
						xValid, yValid, treeNum, depth, lr);
					size_t estCnt = history.getTreesLearnt();
					std::cout << "Real number of trees: " << estCnt << "\n";

					std::vector<Lab_t> residuals = computeResiduals(model);
					curScore = StatisticsHelper::mean(residuals);
					std::cout << "Mean absolute error: " << curScore << "\n";
					std::cout << "Max absolute error: " << StatisticsHelper::maxAbs(residuals) << "\n";
					std::cout << "===================================\n\n\n";

					if (!bestFound || curScore < bestScore) {
						bestFound = true;
						bestScore = curScore;
					}
					++experimentNumber;
				}
			}
		}
	}

	std::cout << "TEST FINISHED\n";
	std::cout << "The best score (MAE) = " << bestScore << "\n";
}


void TestLauncher::singleTestPrint(const size_t treeCount,
	const size_t treeDepth, const size_t binCount,
	const float learnRate, const size_t patience) const {
	std::cout << "Test launched\n";

	std::cout << "Tree count: " << treeCount << "\n";
	std::cout << "Tree depth: " << treeDepth << "\n";
	std::cout << "Bin count: " << binCount << "\n";
	std::cout << "Learning rate: " << learnRate << "\n";
	std::cout << "Patience: " << patience << "\n";
	std::cout << "Fit\n";

	GradientBoosting model(binCount, patience);
	History history = model.fit(xTrain, yTrain,
		xValid, yValid, treeCount, treeDepth, learnRate);
	size_t estCnt = history.getTreesLearnt();
	std::cout << "Real tree count: " << estCnt << "\n";

	std::vector<Lab_t> residuals = computeResiduals(model);
	std::cout << "MAE: " << StatisticsHelper::mean(residuals) << "\n";

	std::cout << "Ensemble:\n";
	model.printModel();

	std::cout << "TEST FINISHED\n";
}


void TestLauncher::printConditions(const size_t treeNum,
	const size_t depth, const size_t binCount,
	const float learningRate) {
	std::cout << "Parameters:\n";
	std::cout << "The number of trees (MAX): " << treeNum << "\n";
	std::cout << "Depth of each tree: " << depth << "\n";
	std::cout << "Bin count for histograms: " << binCount << "\n";
	std::cout << "Learning rate: " << learningRate << "\n";
}

std::vector<Lab_t> TestLauncher::computeResiduals(const GradientBoosting& model) const {
	std::vector<Lab_t> preds;
	for (auto& curTest : xValid) {
		preds.push_back(model.predict(curTest));
	}

	std::cout << "Calculating residuals\n";

	std::vector<Lab_t> residual;
	for (size_t i = 0; i < yValid.size(); ++i) {
		residual.push_back(abs(preds[i] - yValid[i]));
	}
	return residual;
}
