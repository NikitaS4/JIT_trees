#pragma once

#include <vector>
#include <string>

#include "Structs.h"


class DataLoader {
	/*
	X data file format: text file with feature values
	the first number: data count
	the second number: feature count
	remaining numbers: features
	order: s1f1 s1f2 ... s2f1 s2f2 ... where
	sN is sample number, fN is feature number
	
	Y data file format: text file with labels
	the first number: data count
	remaining numbers: labels
	*/
public:
	static std::vector<std::vector<FVal_t>> loadX(const std::string& filename);
	static std::vector<Lab_t> loadY(const std::string& filename);
private:
	DataLoader() = delete;  // static class == helper
};