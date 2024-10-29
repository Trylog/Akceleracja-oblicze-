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
void printRuntimes(int rowsA, int columnsA, int rowsB, int columnsB) {
    auto a = generateRandomMatrix<T>(rowsA, columnsA);
    auto b = generateRandomMatrix<T>(rowsB, columnsB);

    printRuntime<T>(MatrixMultiplication::threadPoolWithBatchingAndQueue<T>, a, b, true,
                    "Thread pool with batching and pre-initialized queue");
    printRuntime<T>(MatrixMultiplication::threadPoolWithBathing<T>, a, b, true,
                    "Thread pool with batching");
}

void testOnRandomMatrix(char datatype, int rowsA, int columnsA, int rowsB, int columnsB) {
    switch (datatype) {
        case '1':
            printRuntimes<short>(rowsA, columnsA, rowsB, columnsB);
            break;
        case '2':
            printRuntimes<int>(rowsA, columnsA, rowsB, columnsB);
            break;
        case '3':
            printRuntimes<long>(rowsA, columnsA, rowsB, columnsB);
            break;
        case '4':
            printRuntimes<long long>(rowsA, columnsA, rowsB, columnsB);
            break;
        case '5':
            printRuntimes<float>(rowsA, columnsA, rowsB, columnsB);
            break;
        case '6':
            printRuntimes<double>(rowsA, columnsA, rowsB, columnsB);
            break;
        case '7':
            printRuntimes<long double>(rowsA, columnsA, rowsB, columnsB);
            break;
        default:
            std::cerr << "Invalid choice!" << std::endl;
    }
}


int getDimensionOfMatrix(std::string message) {
    std::string input;
    std::cout << message;
    std::cin >> input;
    int size = std::stoi(input);

    return size;
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
    int rowsA = getDimensionOfMatrix("Type in rowsA size: ");
    int columnsA = getDimensionOfMatrix("Type in columnsA size: ");
    int rowsB = getDimensionOfMatrix("Type in rowsB size: ");
    int columnsB = getDimensionOfMatrix("Type in columnsB size: ");

    if (columnsA != rowsB) {
        std::cerr << "Matrix1 columns number not equal to Matrix2 rows number." << std::endl;
        return 0;
    }

    char datatype = getDataType();
    testOnRandomMatrix(datatype, rowsA, columnsA, rowsB, columnsB);
    return 0;
}
