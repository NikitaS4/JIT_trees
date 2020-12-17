#ifndef STATISTICS_HELPER_H
#define STATISTICS_HELPER_H

#include "PybindHeader.h"
#include "Structs.h"
#include <vector>

class StatisticsHelper {
public:
	static Lab_t mean(const pytensorY& vals);
	static Lab_t mean(const pytensorY& vals,
		const std::vector<size_t>& idxs);
	static Lab_t maxAbs(const pytensorY& vals);
private:
	StatisticsHelper();
	~StatisticsHelper() = delete;
};

#endif // STATISTICS_HELPER_H
