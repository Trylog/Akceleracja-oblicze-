#pragma once
#include <iostream>
#include <matrixMultiplication.h>
#include <vector>
#include <thread>
// #include <windows.h>
#include <mutex>
#include<condition_variable>

inline std::condition_variable cv;
inline std::mutex cpuCountMutex;

template<typename T>
void multiplySingleColumn(std::vector<std::vector<T> > &result,
                          const std::vector<std::vector<T> > &a, const std::vector<std::vector<T> > &b,
                          int aRow, int bColumn, int numOfElements) {
    T sum = T{}; // T{} is uniform initialization. Return 0 for numeric types

    for (int k = 0; k < numOfElements; ++k) {
        sum += a[aRow][k] * b[k][bColumn];
    }

    result[aRow][bColumn] = sum;
}


namespace MatrixMultiplication {
    template<typename T>
    std::vector<std::vector<T> > singleThread(const std::vector<std::vector<T> > &a,
                                              const std::vector<std::vector<T> > &b) {
        int rowsA = a.size();
        int rowsB = b.size();
        int columnsB = b[0].size();

        std::vector<std::vector<T> > resultMatrix(rowsA, std::vector<T>(columnsB, 0));

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < columnsB; ++j) {
                multiplySingleColumn<T>(resultMatrix, a, b, i, j, rowsB);
            }
        }

        return resultMatrix;
    }


    template<typename T>
    std::vector<std::vector<T> > naiveMultiThreads(const std::vector<std::vector<T> > &a,
                                                   const std::vector<std::vector<T> > &b) {
        int rowsA = a.size();
        int rowsB = b.size();
        int columnsB = b[0].size();

        std::vector<std::vector<T> > resultMatrix(rowsA, std::vector<T>(columnsB, 0));

        std::vector<std::thread> threads;

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < columnsB; ++j) {
                threads.push_back(
                    std::thread(multiplySingleColumn<T>, std::ref(resultMatrix),
                                std::cref(a), std::cref(b),
                                i, j, rowsB)
                );
            }
        }

        for (auto &t: threads) {
            t.join();
        }

        return resultMatrix;
    }

    template <typename T>
std::vector<std::vector<T>> threadPooledMultiThreads(const std::vector<std::vector<T>>& a, const std::vector<std::vector<T>>& b) {
        int rowsA = a.size();
        int rowsB = b.size();
        int columnsB = b[0].size();

        std::vector<std::vector<T>> resultMatrix(rowsA, std::vector<T>(columnsB, 0));

        const unsigned int maxNumberOfCPUCores = std::thread::hardware_concurrency();
        std::cout << "Number of CPU cores: " << maxNumberOfCPUCores << std::endl;

        int currentNumberOfCPUCoresInUse = 0;
        std::mutex mtx;
        std::condition_variable cv;

        std::vector<std::thread> threads;

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < columnsB; ++j) {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return currentNumberOfCPUCoresInUse < maxNumberOfCPUCores; });

                currentNumberOfCPUCoresInUse++;
                std::cout << "Current number of CPU cores in use: " << currentNumberOfCPUCoresInUse << std::endl;

                threads.push_back(
                    std::thread([&, i, j] {
                        multiplySingleColumn<T>(resultMatrix, a, b, i, j, rowsB, currentNumberOfCPUCoresInUse);
                        std::lock_guard<std::mutex> guard(mtx);
                        currentNumberOfCPUCoresInUse--;
                        cv.notify_one();
                    })
                );
            }
        }

        for (auto& t : threads) {
            t.join();
        }

        return resultMatrix;
    }
}
