#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <functional>
#include <limits>
#include <variant>
#include <random>
#include <cstdint> // int8_t
#include<fstream>

#include "matrixMultiplication.h"
#include "avxAlignedVector.h"


template<typename T>
AvxAlignedMatrix<T> generateRandomMatrix(int n, int m) {
    std::random_device rd;
    std::mt19937 gen(rd());
    AvxAlignedMatrix<T> matrix = createAvxAlignedMatrix<T>(n, m);

    T maxValue = 10;
    T minValue = -10;

    // MAX AND MIN LIMITS:
    auto maxIntSigned = 10;
    auto minIntSigned = -10;

    auto maxIntUnsigned = 10;
    int minIntUnsigned = 0;

    auto maxFloatSigned = 10.f;
    auto minFloatSigned = -10.f;

    auto maxFloatUnsigned = 10.f;
    auto minFloatUnsigned = 0.f;

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
            } else if constexpr (std::is_floating_point_v<T>) {
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

template <typename T>
void printMatrix(AvxAlignedMatrix<T> m) {
    for (int i = 0; i < m.size(); ++i) {
        for (int j = 0; j < m[0].size(); ++j) {
            std::cout << m[i][j] << " ";
        }
        std::cout << "\n";
    }
}

template<typename Func, typename... Args>
double measureExecutionTime(Func &&func, Args &&... args) {
    auto start = std::chrono::high_resolution_clock::now();

    auto resultMatrix = func(std::forward<Args>(args)...);

    auto end = std::chrono::high_resolution_clock::now();

    // printMatrix(resultMatrix);

    std::chrono::duration<double, std::micro> duration = end - start;
    return duration.count();
}
template<typename T>
double execTimeSingleThread(AvxAlignedMatrix<T> a, AvxAlignedMatrix<T> b) {
    return measureExecutionTime(MatrixMultiplication::singleThread<T>, a, b, true);
}
template<typename T>
double execTimeSingleThreadSimd(AvxAlignedMatrix<T> a, AvxAlignedMatrix<T> b) {
    return measureExecutionTime(MatrixMultiplication::AVX_singleThread<T>, a, b, true);
}
template<typename T>
double execTimeNaiveMultiThreads(AvxAlignedMatrix<T> a, AvxAlignedMatrix<T> b) {
    return measureExecutionTime(MatrixMultiplication::naiveMultiThreads<T>, a, b, true);
}
template<typename T>
double execTimeThreadPoolWithBatching(AvxAlignedMatrix<T> a, AvxAlignedMatrix<T> b) {
    return measureExecutionTime(MatrixMultiplication::threadPoolWithBatching<T>, a, b, true);
}
template<typename T>
double execTimeThreadPoolWithBatchingAndQueue(AvxAlignedMatrix<T> a, AvxAlignedMatrix<T> b) {
    return measureExecutionTime(MatrixMultiplication::threadPoolWithBatchingAndQueue<T>, a, b, true);
}
template<typename T>
double execTimeThreadPoolWithBatchingAndQueueSimd(AvxAlignedMatrix<T> a, AvxAlignedMatrix<T> b) {
    return measureExecutionTime(MatrixMultiplication::AVX_threadPoolWithBatchingAndQueue<T>, a, b, true);
}


template<typename T>
void printRuntimes(int rowsA, int columnsA, int rowsB, int columnsB) {
    auto a = generateRandomMatrix<T>(rowsA, columnsA);
    auto b = generateRandomMatrix<T>(rowsB, columnsB);

    std::ofstream resultFile("result_" + std::to_string(rowsA) + "_" + typeid(T).name() + ".csv", std::ios::app);

    if(rowsA < 1000) {
        resultFile << "Single thread;";
        double exec_time_single = execTimeSingleThread<T>(a, b);
        resultFile << exec_time_single << std::endl;

        resultFile << "Single thread SIMD;";
        double exec_time_single_simd = execTimeSingleThreadSimd<T>(a, b);
        resultFile << exec_time_single_simd << std::endl;

        resultFile<< "Naive multi threads;";
        double exec_time_naive_multi = execTimeNaiveMultiThreads<T>(a, b);
        resultFile << exec_time_naive_multi << std::endl;
    }

    if(rowsA < 12500) {
        resultFile << "Thread pool with batching;";
        double exec_time_thread_pool_batching = execTimeThreadPoolWithBatching<T>(a, b);
        resultFile << exec_time_thread_pool_batching << std::endl;
    }

    resultFile << "Thread pool with batching and queue;";
    double exec_time_thread_pool_batching_queue = execTimeThreadPoolWithBatchingAndQueue<T>(a, b);
    resultFile << exec_time_thread_pool_batching_queue << std::endl;

    resultFile << "Thread pool with batching and queue SIMD;";
    double exec_time_thread_pool_batching_queue_simd = execTimeThreadPoolWithBatchingAndQueueSimd<T>(a, b);
    resultFile << exec_time_thread_pool_batching_queue_simd << std::endl;

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
    // int rowsA = getDimensionOfMatrix("Type in rowsA size: ");
    // int columnsA = getDimensionOfMatrix("Type in columnsA size: ");
    // int rowsB = getDimensionOfMatrix("Type in rowsB size: ");
    // int columnsB = getDimensionOfMatrix("Type in columnsB size: ");
    //
    // if (columnsA != rowsB) {
    //     std::cerr << "Matrix1 columns number not equal to Matrix2 rows number." << std::endl;
    //     exit(3);
    // }

    std::queue<int> volumes_queue;

    std::vector<int> values = {10, 50, 100, 250, 1000, 2500, 5000, 10000, 12500, 15000, 17500, 20000, 25000};
    for (int i = 0; i < values.size(); i++) {
        volumes_queue.push(values[i]);
    }

    std::cout << "Starting tests..." << std::endl;

    while (!volumes_queue.empty()) {
        std::cout << "Volume: " << volumes_queue.front() << std::endl;
        int volume = volumes_queue.front();
        volumes_queue.pop();
        int rowsA = volume;
        int columnsA = volume;
        int rowsB = volume;
        int columnsB = volume;

        char datatype = getDataType();
        testOnRandomMatrix(datatype, rowsA, columnsA, rowsB, columnsB);
    }
}


int main() {
    runWithGeneratedMatrices();

    return 0;
}
