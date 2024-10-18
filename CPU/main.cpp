#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <functional>
#include <limits>
#include <variant>

#include "matrixMultiplication.h"

template <typename T>
std::vector<std::vector<T>> generateRandomMatrix(int n, int m) {
    std::srand(static_cast<unsigned>(std::time(0)));
    std::vector<std::vector<T>> matrix(n, std::vector<T>(m));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if constexpr (std::is_integral_v<T>) {
                // Generate a random integer number
                matrix[i][j] = static_cast<T>(std::rand() % std::numeric_limits<T>::max());
            } else if constexpr (std::is_floating_point_v<T>) {
                // Generate a floating-point number normalized to the range [0, max]
                matrix[i][j] = static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX) * std::numeric_limits<T>::max();
            }
        }
    }

    return matrix;
}


template <typename Func, typename... Args>
double measureExecutionTime(Func&& func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();

    func(std::forward<Args>(args)...);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> duration = end - start;
    return duration.count();
}


template <typename T>
void printRuntimes(int n, int m) {
    auto a = generateRandomMatrix<T>(n, m);
    auto b = generateRandomMatrix<T>(n, m);

    double duration = measureExecutionTime(MatrixMultiplication::singleThread<T>, a, b);
    std::cout << "Single-threaded Time: " << duration << " microseconds" << std::endl;

    double duration2 = measureExecutionTime(MatrixMultiplication::naiveMultiThreads<T>, a, b);
    std::cout << "Naive multi-threaded Time: " << duration2 << " microseconds" << std::endl;
}


int main()
{
    std::string input;
    std::cout << "Type in size of matrices: ";
    std::cin >> input;
    int N = std::stoi(input);

    char datatype;
    std::cout << "Choose datatype:\n";
    std::cout <<
        "1. int\n" <<
        "2. float\n" <<
        "3. double\n";
    std::cin >> datatype;

    switch (datatype) {
        case '1':
            printRuntimes<int>(N, N);
            break;
        case '2':
            printRuntimes<float>(N, N);
            break;
        case '3':
            printRuntimes<double>(N, N);
            break;
        default:
            std::cerr << "Invalid choice!" << std::endl;
    }

    return 0;
}
