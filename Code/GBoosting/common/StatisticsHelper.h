#ifndef STATISTICS_HELPER_H
#define STATISTICS_HELPER_H

#include "Structs.h"
#include <vector>

class StatisticsHelper {
public:
	static Lab_t mean(const std::vector<Lab_t>& vals);
	static Lab_t mean(const std::vector<Lab_t>& vals,
		const std::vector<size_t>& idxs);
	static Lab_t maxAbs(const std::vector<Lab_t>& vals);
private:
	StatisticsHelper();
	~StatisticsHelper() = delete;
};

#endif // STATISTICS_HELPER_H
