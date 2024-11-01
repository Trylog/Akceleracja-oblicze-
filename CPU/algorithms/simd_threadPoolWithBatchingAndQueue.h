#pragma once

#include <immintrin.h> // AVX
#include <vector>
#include <cstdint>
#include <functional>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "avxAlignedVector.h"
#include "algorithms/_helperFunctions.h"
#include "algorithms/_simdMultiplicationFunctions.h"


template<typename T>
void avx2_threadPoolWithBatchingAndQueueWorker(AvxAlignedMatrix<T> &resultMatrix,
                                               const AvxAlignedMatrix<T> &a,
                                               const AvxAlignedMatrix<T> &b,
                                               int numOfElements,
                                               std::queue<std::pair<int, int>>& tasks,
                                               std::mutex& mtx, std::condition_variable& cv,
                                               bool& done,
                                               std::function<void(const AvxAlignedMatrix<T>&,
                                                                  const AvxAlignedMatrix<T>&,
                                                                  AvxAlignedMatrix<T>&,
                                                                  int, int, int, int)> multiplyFunc) {
    const int simdSizeBytes = 32;
    const int numOfVariablesPerSimd = simdSizeBytes / sizeof(T);
    int numOfElementsThatFitSimd = numOfElements / numOfVariablesPerSimd;
    numOfElementsThatFitSimd *= numOfVariablesPerSimd;

    while (true) {
        std::pair<int, int> task;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]() { return done || !tasks.empty(); });

            if (done && tasks.empty()) return;

            task = tasks.front();
            tasks.pop();
        }

        int aIndex = task.first;
        int bIndex = task.second;
        multiplyFunc(a, b, resultMatrix, aIndex, bIndex, numOfElements, numOfElementsThatFitSimd);
    }
}


template <typename T>
AvxAlignedMatrix<T> AVX_threadPoolWithBatchingAndQueue(const AvxAlignedMatrix<T> &a,
                                                       const AvxAlignedMatrix<T> &b) {
    int rowsA = a.size();
    int rowsB = b.size();
    int columnsB = b[0].size();
    int numOfElements = rowsB;

    AvxAlignedMatrix<T> resultMatrix = createAvxAlignedMatrix<T>(rowsA, columnsB);
    const AvxAlignedMatrix<T> &newB = transposeMatrix(b);

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

    // Define multiplyFunc based on type
    std::function<void(const AvxAlignedMatrix<T>&,
                       const AvxAlignedMatrix<T>&,
                       AvxAlignedMatrix<T>&,
                       int, int, int, int)> multiplyFunc;

    if constexpr (std::is_same_v<T, float>) {
        multiplyFunc = multiplySingleResultFloat;
    } else if constexpr (std::is_same_v<T, double>) {
        multiplyFunc = multiplySingleResultDouble;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        multiplyFunc = multiplySingleResultInt8;
    } else if constexpr (std::is_same_v<T, int>) {
        multiplyFunc = multiplySingleResultInt;
    }

    // Create thread pool
    std::vector<std::thread> threads;
    for (unsigned int n = 0; n < maxNumberOfCPUCores; ++n) {
        threads.emplace_back([&]() {
            avx2_threadPoolWithBatchingAndQueueWorker<T>(
                    resultMatrix, a, newB, numOfElements, tasks, mtx, cv, done, multiplyFunc);
        });
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
