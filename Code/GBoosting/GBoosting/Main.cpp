#include <iostream>
#include <vector>

#include "GBoosting.h"
#include "GBTest.h"
#include "StatisticsHelper.h"

void testSort() {
	std::vector<FVal_t> sample = { 1, 3, 8, 15, 2, 16, 10, 7, 4, 3 };
	std::vector<size_t> sorted = GBTest::testSort(sample);

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
}

float randInterval(float from, float to) {
	if (to <= from)
		throw std::runtime_error("Wrong args passed to randInterval");
	return float(from + (double(to) - from) * double(rand()) / (double(RAND_MAX) + 1));
}

void testBoosting() {
	size_t treeCount = 3;

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

	GradientBoosting model;

	std::cout << "Fitting model\n";
	model.fit(xTrain, yTrain, treeCount);

	std::cout << "Model has been fit. Collecting predictions\n";

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

int main() {
	try {
		//testSort();
		testBoosting();
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