#include <iostream>
#include <vector>

#include "GBoosting.h"
#include "GBTest.h"
#include "StatisticsHelper.h"
#include "TestLauncher.h"
#include "History.h"

void testSort() {
	std::vector<FVal_t> sample = { 1, 3, 8, 15, 2, 16, 10, 7, 4, 3 };
	std::vector<size_t> backIdxs;
	std::vector<size_t> sorted = GBTest::testSort(sample, backIdxs);

	std::cout << "Original array:\n";
	for (auto& it : sample) {
		std::cout << it << " ";
	}
	std::cout << "\nSorted array:\n";
	for (auto& it : sorted) {
		std::cout << sample[it] << " ";
	}
	std::cout << "\nIndexes:\n";
	for (auto& it : sorted) {
		std::cout << it << " ";
	}
	std::cout << "\nBack indexes:\n";
	for (auto& it : backIdxs) {
		std::cout << it << " ";
	}
}

float randInterval(float from, float to) {
	if (to <= from)
		throw std::runtime_error("Wrong args passed to randInterval");
	return float(from + (double(to) - from) * double(rand()) / (double(RAND_MAX) + 1));
}

void testBoosting() {
	size_t treeCount = 3;
	size_t treeDepth = 2;
	size_t binCount = 256;
	float learningRate = 0.5;

	// task: regression
	// 2 features
	// function: f(x1, x2) = x1 ** 2 + x2 + 10sin(x3) + 30

	size_t trainLen = 1000;
	size_t testLen = 100;
	
	// fill train dataset
	std::vector<std::vector<FVal_t>> xTrain;
	std::vector<Lab_t> yTrain;
	FVal_t x1, x2, x3;
	for (size_t i = 0; i < trainLen; ++i) {
		x1 = randInterval(-5, 5);
		x2 = randInterval(-5, 5);
		x3 = randInterval(-5, 5);
		xTrain.push_back({ x1, x2, x3 });
		yTrain.emplace_back(x1 * x1 + x2 + 10 * sin(x3) + 30);
	}

	// fill test dataset
	std::vector<std::vector<FVal_t>> xTest;
	std::vector<Lab_t> yTest;
	for (size_t i = 0; i < testLen; ++i) {
		x1 = randInterval(-5, 5);
		x2 = randInterval(-5, 5);
		x3 = randInterval(-5, 5);
		xTest.push_back({ x1, x2, x3 });
		yTest.emplace_back(x1 * x1 + x2 + 10 * sin(x3) + 30);
	}

	std::cout << "Dataset generated\n";

	GradientBoosting model(binCount);

	std::cout << "Fitting model\n";
	History history = model.fit(xTrain, yTrain, xTest, yTest, 
		treeCount, treeDepth, learningRate);
	size_t estimCount = history.getTreesLearnt();

	std::cout << "Model has been fit. Estimators count = " << estimCount << "\n";

	std::vector<Lab_t> preds;
	for (auto& curTest : xTest) {
		preds.push_back(model.predict(curTest));
	}

	std::cout << "Calculating residuals\n";

	std::vector<Lab_t> residual;
	for (size_t i = 0; i < testLen; ++i) {
		residual.push_back(abs(preds[i] - yTest[i]));
	}

	std::cout << "Mean absolute error: " << StatisticsHelper::mean(residual) << "\n";
	std::cout << "Max absolute error: " << StatisticsHelper::maxAbs(residual) << "\n";
}

void testGrid() {
	std::string xTrainFile("input\\xTr.txt");
	std::string yTrainFile("input\\yTr.txt");
	std::string xValidFile("input\\xV.txt");
	std::string yValidFile("input\\yV.txt");

	TestLauncher testLauncher(xTrainFile, yTrainFile,
		xValidFile, yValidFile);

	std::vector<size_t> treeCounts({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
	std::vector<size_t> treeDepths({1, 2, 3});
	std::vector<size_t> binCounts({4, 16, 32});
	std::vector<float> learnRates({1.0f, 0.5f, 0.8f, 1.0f});
	size_t patience = 10;

	testLauncher.performTest(treeCounts, treeDepths, binCounts,
		learnRates, patience);
}

void testSingle() {
	std::string xTrainFile("input\\xTr.txt");
	std::string yTrainFile("input\\yTr.txt");
	std::string xValidFile("input\\xV.txt");
	std::string yValidFile("input\\yV.txt");

	TestLauncher testLauncher(xTrainFile, yTrainFile,
		xValidFile, yValidFile);

	size_t treeCount = 50;
	size_t treeDepth = 3;
	size_t binCount = 256;
	float learnRate = 0.4f;
	size_t patience = 4;
	testLauncher.singleTestPrint(treeCount, treeDepth,
		binCount, learnRate, patience);
}

void testTrivial() {
	std::vector<std::vector<FVal_t>> x = { {0.1f}, {0.3f}, 
	{0.4f}, {0.5f}, {0.6f}, {0.7f}, {0.8f}, {1.0f} };
	std::vector<Lab_t> y = {0.1f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
	1.0f};
	std::vector<std::vector<FVal_t>> xVal = { {0.2f}, {0.9f} };
	std::vector<Lab_t> yVal = { 0.2f, 0.9f };

	TestLauncher testLauncher(x, y, xVal, yVal);

	size_t treeCount = 4;
	size_t treeDepth = 1;
	size_t binCount = 16;
	float learnRate = 1.0f;
	size_t patience = 4;
	testLauncher.singleTestPrint(treeCount, treeDepth,
		binCount, learnRate, patience);
}

int main() {
	try {
		//testSort();
		//testBoosting();
		//testGrid();
		testSingle();
		//testTrivial();
	}
	catch (std::runtime_error & err) {
		std::cout << "Exception caught (re): " << err.what() << "\n";
	}
	catch (std::exception & ex) {
		std::cout << "Exception caught: " << ex.what() << "\n";
	}
	catch (...) {
		std::cout << "Unknown exception caught\n";
	}
	
	return 0;
}