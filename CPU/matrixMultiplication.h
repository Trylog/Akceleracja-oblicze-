#pragma once
#include <iostream>
#include <matrixMultiplication.h>
#include <vector>
#include <thread>
#include <iomanip>
// #include <windows.h>
#include <mutex>
#include<condition_variable>
#include<queue>


template<typename T>
void multiplySingleColumn(std::vector<std::vector<T> > &result,
                          const std::vector<std::vector<T> > &a, const std::vector<std::vector<T> > &b,
                          int aIndex, int bIndex, int numOfElements, bool wasTransposed = false) {
    T sum = T{}; // T{} is uniform initialization. Return 0 for numeric types

    for (int i = 0; i < numOfElements; ++i) {
        if (wasTransposed) {
            sum += a[aIndex][i] * b[bIndex][i];
        }
        else{
            sum += a[aIndex][i] * b[i][bIndex];
        }
    }

    result[aIndex][bIndex] = sum;
}


template<typename T>
std::vector<std::vector<T>> transposeMatrix(const std::vector<std::vector<T>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<T>> transposed(cols, std::vector<T>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}


template<typename T>
void threadPoolWithBatchingWorker(std::vector<std::vector<T>>& resultMatrix,
                                  const std::vector<std::vector<T> >& a, const std::vector<std::vector<T>>& b, int numOfElements,
                                  std::queue<std::pair<int, int>>& tasks,
                                  std::mutex& mtx, std::condition_variable& cv,
                                  bool& done) {
    while (true) {
        std::pair<int, int> task;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]() { return done || !tasks.empty(); });

            if (done && tasks.empty()) return;

            task = tasks.front();
            tasks.pop();
        }

        int i = task.first;
        int j = task.second;
        multiplySingleColumn(resultMatrix, a, b,
                             i, j, numOfElements,
                             true);
    }
}


namespace MatrixMultiplication {
    template<typename T>
    std::vector<std::vector<T> > singleThread(const std::vector<std::vector<T> > &a,
                                              const std::vector<std::vector<T> > &b,
                                              bool withTransposition) {
        int rowsA = a.size();
        int rowsB = b.size();
        int columnsB = b[0].size();
        int numOfElements = rowsB;

        std::vector<std::vector<T> > resultMatrix(rowsA, std::vector<T>(columnsB, 0));
        const std::vector<std::vector<T>>& newB = withTransposition ? transposeMatrix(b) : b;

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < columnsB; ++j) {
                multiplySingleColumn<T>(resultMatrix, a, newB,
                                        i, j, numOfElements,
                                        false
                );
            }
        }

        return resultMatrix;
    }


    template<typename T>
    std::vector<std::vector<T> > naiveMultiThreads(const std::vector<std::vector<T> > &a,
                                                   const std::vector<std::vector<T> > &b,
                                                   bool withTransposition) {
        int rowsA = a.size();
        int rowsB = b.size();
        int columnsB = b[0].size();
        int numOfElements = rowsB;

        std::vector<std::vector<T> > resultMatrix(rowsA, std::vector<T>(columnsB, 0));
        const std::vector<std::vector<T>>& newB = withTransposition ? transposeMatrix(b) : b;

        std::vector<std::thread> threads;

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < columnsB; ++j) {
                threads.push_back(
                        std::thread(multiplySingleColumn<T>,
                                    std::ref(resultMatrix), std::cref(a), std::cref(newB),
                                    i, j, numOfElements,
                                    false
                        )
                );
            }
        }

        for (auto &t: threads) {
            t.join();
        }

        return resultMatrix;
    }


    template <typename T>
    std::vector<std::vector<T>> threadPoolWithBatching(const std::vector<std::vector<T>>& a,
                                                       const std::vector<std::vector<T>>& b,
                                                       bool withTransposition) {
        int rowsA = a.size();
        int rowsB = b.size();
        int columnsB = b[0].size();
        int numOfElements = rowsB;

        std::vector<std::vector<T>> resultMatrix(rowsA, std::vector<T>(columnsB, 0));
        const std::vector<std::vector<T>>& newB = withTransposition ? transposeMatrix(b) : b;

        const unsigned int maxNumberOfCPUCores = std::thread::hardware_concurrency();

        std::queue<std::pair<int, int>> tasks;
        std::mutex mtx;
        std::condition_variable cv;
        bool done = false;

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < columnsB; ++j) {
                tasks.emplace(i, j);
            }
        }

        std::vector<std::thread> threads;
        for (unsigned int n = 0; n < maxNumberOfCPUCores; ++n) {
            threads.emplace_back(threadPoolWithBatchingWorker<T>,
                                 std::ref(resultMatrix), std::cref(a), std::cref(b), numOfElements,
                                 std::ref(tasks), std::ref(mtx), std::ref(cv), std::ref(done)
            );
        }

        cv.notify_all();

        {
            std::unique_lock<std::mutex> lock(mtx);
            done = true;
        }
        cv.notify_all();
        for (auto& thread : threads) {
            thread.join();
        }

        return resultMatrix;
    }
}
