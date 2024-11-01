#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <functional>
#include <limits>
#include <variant>
#include <random>
#include <cstdint> // int8_t

#include "matrixMultiplication.h"


template <typename T>
std::vector<std::vector<T>> generateRandomMatrix(int n, int m) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::vector<T>> matrix(n, std::vector<T>(m));

    T maxValue = 100;
    T minValue = -100;

        // MAX AND MIN LIMITS:
        auto maxIntSigned = std::numeric_limits<T>::max();
        auto minIntSigned = std::numeric_limits<T>::min();

        auto maxIntUnsigned = std::numeric_limits<T>::max();
        int minIntUnsigned = 0;

        auto maxFloatSigned = std::numeric_limits<T>::max();
        auto minFloatSigned = -std::numeric_limits<T>::max();

        auto maxFloatUnsigned = std::numeric_limits<T>::max();
        auto minFloatUnsigned = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if constexpr (std::is_integral_v<T>) {
                if constexpr (std::is_signed_v<T>) {
                    std::uniform_int_distribution<T> dist(minIntSigned, maxIntSigned);
                    matrix[i][j] = dist(gen);
                } else {
                    std::uniform_int_distribution<T> dist(minIntUnsigned, maxIntUnsigned);
                    matrix[i][j] = dist(gen);
                }
            }

            else if constexpr (std::is_floating_point_v<T>) {
                if constexpr (std::is_signed_v<T>) {
                    std::uniform_real_distribution<T> dist(minFloatSigned, maxFloatSigned);
                    matrix[i][j] = dist(gen);
                } else {
                    std::uniform_real_distribution<T> dist(minFloatUnsigned, maxFloatUnsigned);
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


template <typename T>
void printRuntimes(int rowsA, int columnsA, int rowsB, int columnsB) {
    auto a = generateRandomMatrix<T>(rowsA, columnsA);
    auto b = generateRandomMatrix<T>(rowsB, columnsB);
    double duration;

    ///*
    duration = measureExecutionTime(MatrixMultiplication::AVX_threadPoolWithBatchingAndQueue<T>, a, b);
    std::cout << "AVX2 thread pool with batching and queue, transposition=true: " << std::fixed << std::setprecision(2) <<
              duration << " microseconds" << std::endl;
    //*/

    ///*
    duration = measureExecutionTime(MatrixMultiplication::AVX_singleThread<T>, a, b);
    std::cout << "AVX2 single thread, transposition=true: " << std::fixed << std::setprecision(2) <<
              duration << " microseconds" << std::endl;
    //*/

    /*
    duration = measureExecutionTime(MatrixMultiplication::threadPoolWithBatchingAndQueue<T>, a, b, true);
    std::cout << "Thread pool with batching and queue, transposition=true: " << std::fixed << std::setprecision(2) <<
        duration << " microseconds" << std::endl;
    */
}


void testOnRandomMatrix(char datatype, int rowsA, int columnsA, int rowsB, int columnsB) {
    switch (datatype) {
        case '1':
            printRuntimes<int8_t>(rowsA, columnsA, rowsB, columnsB);
            break;
        case '2':
            printRuntimes<int>(rowsA, columnsA, rowsB, columnsB);
            break;
        case '3':
            printRuntimes<float>(rowsA, columnsA, rowsB, columnsB);
            break;
        case '4':
            printRuntimes<double>(rowsA, columnsA, rowsB, columnsB);
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
              "1. int8_t: 8 bits\n" <<
              "2. int: 32 bits\n" <<
              "3. float: 32 bits bits\n" <<
              "4. double: 64 bits\n";
    std::cout << "Choice: ";
    std::cin >> datatype;

    return datatype;
}


void runWithGeneratedMatrices() {
    int rowsA = getDimensionOfMatrix("Type in rowsA size: ");
    int columnsA = getDimensionOfMatrix("Type in columnsA size: ");
    int rowsB = getDimensionOfMatrix("Type in rowsB size: ");
    int columnsB = getDimensionOfMatrix("Type in columnsB size: ");

    if (columnsA != rowsB) {
        std::cerr << "Matrix1 columns number not equal to Matrix2 rows number." << std::endl;
        exit(3);
    }

    char datatype = getDataType();
    testOnRandomMatrix(datatype, rowsA, columnsA, rowsB, columnsB);
}


int main() {
    runWithGeneratedMatrices();
    return 0;
}
