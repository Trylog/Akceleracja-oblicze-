#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>

#include "matrixOperations.h"


std::vector<std::vector<int>> generateMatrix(int n, int m, int maxNumber) {
    std::srand(std::time(0));

    std::vector<std::vector<int>> matrix(n, std::vector<int>(m));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            matrix[i][j] = std::rand() % (maxNumber + 1);
        }
    }

    return matrix;
}


int main()
{
    int N = 100;
    auto a = generateMatrix(N, N, INT_MAX);
    auto b = generateMatrix(N, N, INT_MAX);

    // Measure time for single-threaded multiplication
    auto start = std::chrono::high_resolution_clock::now();
    auto res1 = MatrixOps::multiplyOnSingleThread(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start; // Duration in microseconds
    std::cout << "Single-threaded Time: " << duration.count() << " microseconds" << std::endl;

    // Measure time for multi-threaded multiplication
    auto start2 = std::chrono::high_resolution_clock::now();
    auto res2 = MatrixOps::multiply(a, b);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration2 = end2 - start2; // Duration in microseconds
    std::cout << "Multi-threaded Time: " << duration2.count() << " microseconds" << std::endl;
}
