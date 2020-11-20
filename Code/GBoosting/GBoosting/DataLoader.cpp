#include "DataLoader.h"
#include <fstream>
#include <stdexcept>


std::vector<std::vector<FVal_t>> DataLoader::loadX(const std::string& filename) {
	std::vector<std::vector<FVal_t>> xData;

	std::ifstream infile(filename, std::ios::in);
	if (!infile.good())
		throw std::runtime_error("Can't open file " + filename);
	
	try {
		size_t featureCount;
		size_t dataCount;

		infile >> dataCount >> featureCount;
		for (size_t sample = 0; sample < dataCount; ++sample) {
			std::vector<FVal_t> sampleRepr;
			FVal_t featureVal;
			for (size_t feature = 0; feature < featureCount; ++feature) {
				infile >> featureVal;
				sampleRepr.push_back(featureVal);
			}
			xData.push_back(sampleRepr);
		}
	}
	catch (std::runtime_error& err) {
		infile.close();  // close the file and throw further
		throw err;
	}

	infile.close();  // close the file
	return xData;
}

std::vector<Lab_t> DataLoader::loadY(const std::string& filename) {
	std::vector<Lab_t> yData;

	std::ifstream infile(filename, std::ios::in);
	if (!infile.good())
		throw std::runtime_error("Can't open file " + filename);

	try {
		size_t dataCount;
		infile >> dataCount;
		Lab_t curLabel;
		for (size_t sample = 0; sample < dataCount; ++sample) {
			infile >> curLabel;
			yData.push_back(curLabel);
		}
	}
	catch (std::runtime_error & err) {
		infile.close();  // close the file and throw further
		throw err;
	}

	infile.close();  // close the file
	return yData;
}