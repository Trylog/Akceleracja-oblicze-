#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <functional>

#include "matrixMultiplication.h"


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


template <typename Func, typename... Args>
double measureExecutionTime(Func&& func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();

    // Call the function with the provided arguments
    std::forward<Func>(func)(std::forward<Args>(args)...);

    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::micro> duration = end - start;
    return duration.count(); // Return duration in microseconds
}


int main()
{
    int N = 100;
    auto a = generateMatrix(N, N, INT_MAX);
    auto b = generateMatrix(N, N, INT_MAX);

    auto duration = measureExecutionTime(MatrixMultiplication::singleThread, a, b);
    std::cout << "Single-threaded Time: " << duration.count() << " microseconds" << std::endl;

    auto duration2 = measureExecutionTime(MatrixMultiplication::naiveMultiThreads, a, b);
    std::cout << "Multi-threaded Time: " << duration2.count() << " microseconds" << std::endl;
}
