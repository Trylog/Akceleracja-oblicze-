#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <functional>
#include <limits>
#include <variant>
#include <random>

#include "matrixMultiplication.h"


template <typename T>
std::vector<std::vector<T>> generateRandomMatrix(int n, int m) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::vector<T>> matrix(n, std::vector<T>(m));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if constexpr (std::is_integral_v<T>) {
                if constexpr (std::is_signed_v<T>) {
                    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
                    matrix[i][j] = dist(gen);
                } else {
                    std::uniform_int_distribution<T> dist(0, std::numeric_limits<T>::max());
                    matrix[i][j] = dist(gen);
                }
            } else if constexpr (std::is_floating_point_v<T>) {
                if constexpr (std::is_signed_v<T>) {
                    std::uniform_real_distribution<T> dist(-std::numeric_limits<T>::max(), std::numeric_limits<T>::max());
                    matrix[i][j] = dist(gen);
                } else {
                    std::uniform_real_distribution<T> dist(0, std::numeric_limits<T>::max());
                    matrix[i][j] = dist(gen);
                }
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


template <typename T, typename Func>
void printRuntime(Func func,
                  const std::vector<std::vector<T>>& a,
                  const std::vector<std::vector<T>>& b,
                  bool transpose,
                  const std::string& description) {
    double duration = measureExecutionTime(func, a, b, transpose);
    std::cout << description << ", transpose=" << transpose << " "
              << std::fixed << std::setprecision(2) << duration << " microseconds" << std::endl;
}


template <typename T>
void printRuntimes(int n, int m) {
    auto a = generateRandomMatrix<T>(n, m);
    auto b = generateRandomMatrix<T>(n, m);

    // Now directly passing the function pointers without lambdas
    printRuntime<T>(MatrixMultiplication::threadPoolWithBatching<T>, a, b, true,
                    "Thread pool with batching");
    printRuntime<T>(MatrixMultiplication::naiveMultiThreads<T>, a, b, true, "Naive multi-threading");
}

void testOnRandomMatrix(char datatype, int n, int m) {
    switch (datatype) {
        case '1':
            printRuntimes<short>(n, m);
            break;
        case '2':
            printRuntimes<int>(n, m);
            break;
        case '3':
            printRuntimes<long>(n, m);
            break;
        case '4':
            printRuntimes<long long>(n, m);
            break;
        case '5':
            printRuntimes<float>(n, m);
            break;
        case '6':
            printRuntimes<double>(n, m);
            break;
        case '7':
            printRuntimes<long double>(n, m);
            break;
        default:
            std::cerr << "Invalid choice!" << std::endl;
    }
}


int getDimensionOfMatrix() {
    std::string input;
    std::cout << "Type in size of matrices: ";
    std::cin >> input;
    int N = std::stoi(input);

    return N;
}


char getDataType() {
    char datatype;
    std::cout << "Choose datatype:\n";
    std::cout <<
              "1. short\n" <<
              "2. int\n" <<
              "3. long\n" <<
              "4. long long\n" <<
              "5. float\n" <<
              "6. double\n" <<
              "7. long double\n";
    std::cout << "Choice: ";
    std::cin >> datatype;

    return datatype;
}


int main() {
    int N = getDimensionOfMatrix();
    char datatype = getDataType();
    testOnRandomMatrix(datatype, N, N);
    return 0;
}
