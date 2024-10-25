#pragma once
#include <iostream>
#include <matrixMultiplication.h>
#include <vector>
#include <thread>
// #include <windows.h>
#include <mutex>
#include<condition_variable>
#include<queue>

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

template<typename T>
void multiplyTransposed(std::vector<std::vector<T> > &result,
                        const std::vector<std::vector<T> > &a, const std::vector<std::vector<T> > &b,
                        int aRow, int bRow, int numOfElements) {
    T sum = T{}; // T{} is uniform initialization. Return 0 for numeric types

    for (int k = 0; k < numOfElements; ++k) {
        sum += a[aRow][k] * b[bRow][k];
    }

    result[aRow][bRow] = sum;
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

        // Task queue
        std::queue<std::pair<int, int>> tasks;
        std::mutex mtx;
        std::condition_variable cv;
        bool done = false;

        // Populate tasks queue with all element positions to compute
        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < columnsB; ++j) {
                tasks.emplace(i, j);
            }
        }

        // Function to be run by each thread, picking up tasks from the queue
        auto worker = [&]() {
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
                multiplySingleColumn(resultMatrix, a, b, i, j, rowsB);
            }
        };

        // Launch a fixed number of threads
        std::vector<std::thread> threads;
        for (unsigned int n = 0; n < maxNumberOfCPUCores; ++n) {
            threads.emplace_back(worker);
        }

        // Notify all threads to start processing
        cv.notify_all();

        // Join all threads after work is done
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

    template <typename T>
std::vector<std::vector<T>> threadPooledMultiThreadsTransposed(const std::vector<std::vector<T>>& a, const std::vector<std::vector<T>>& b) {
        int rowsA = a.size();
        int rowsB = b.size();
        int columnsB = b[0].size();

        std::vector<std::vector<T>> resultMatrix(rowsA, std::vector<T>(columnsB, 0));
        std::vector<std::vector<T>> bTransposed = transposeMatrix(b);

        const unsigned int maxNumberOfCPUCores = std::thread::hardware_concurrency();
        std::cout << "Number of CPU cores: " << maxNumberOfCPUCores << std::endl;

        // Task queue
        std::queue<std::pair<int, int>> tasks;
        std::mutex mtx;
        std::condition_variable cv;
        bool done = false;

        // Populate tasks queue with all element positions to compute
        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < columnsB; ++j) {
                tasks.emplace(i, j);
            }
        }

        // Function to be run by each thread, picking up tasks from the queue
        auto worker = [&]() {
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
                multiplyTransposed(resultMatrix, a, bTransposed, i, j, rowsB);
            }
        };

        // Launch a fixed number of threads
        std::vector<std::thread> threads;
        for (unsigned int n = 0; n < maxNumberOfCPUCores; ++n) {
            threads.emplace_back(worker);
        }

        // Notify all threads to start processing
        cv.notify_all();

        // Join all threads after work is done
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
