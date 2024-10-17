#pragma once
#include <vector>

namespace MatrixMultiplication {
	std::vector<std::vector<int>> singleThread(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b);
	std::vector<std::vector<int>> naiveMultiThreads(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b);
}