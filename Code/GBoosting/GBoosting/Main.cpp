#include <iostream>
#include <vector>

#include "GBoosting.h"
#include "GBTest.h"

void testSort() {
	std::vector<FVal_t> sample = { 1, 3, 8, 15, 2, 16, 10, 7, 4, 3 };
	std::vector<size_t> sorted;
	GBTest::testSort(sample, sorted);

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

int main() {
	try {
		testSort();
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