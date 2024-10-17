#pragma once
#include <vector>

namespace MatrixOps {
	std::vector<std::vector<int>> multiply(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b);
	std::vector<std::vector<int>> multiplyOnSingleThread(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b);
}