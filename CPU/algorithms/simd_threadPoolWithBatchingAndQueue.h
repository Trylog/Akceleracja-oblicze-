#pragma once

#include <immintrin.h> // AVX
#include <vector>
#include <cstdint>
#include <functional>

#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "algorithms/_helperFunctions.h"


using MultiplyFunctionFloat = void(*)(const std::vector<std::vector<float>>&,
                                      const std::vector<std::vector<float>>&,
                                      std::vector<std::vector<float>>&,
                                      int, int, int);

using MultiplyFunctionDouble = void(*)(const std::vector<std::vector<double>>&,
                                       const std::vector<std::vector<double>>&,
                                       std::vector<std::vector<double>>&,
                                       int, int, int);

using MultiplyFunctionInt8 = void(*)(const std::vector<std::vector<int8_t>>&,
                                     const std::vector<std::vector<int8_t>>&,
                                     std::vector<std::vector<int8_t>>&,
                                     int, int, int);

using MultiplyFunctionInt = void(*)(const std::vector<std::vector<int>>&,
                                    const std::vector<std::vector<int>>&,
                                    std::vector<std::vector<int>>&,
                                    int, int, int);


void multiplySingleResultFloat(const std::vector<std::vector<float>>& a,
                               const std::vector<std::vector<float>>& bT,
                               std::vector<std::vector<float>>& resultMatrix,
                               int aIndex, int bIndex, int numOfElements) {
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 vec_a;
    __m256 vec_b;

    for (int i = 0; i < numOfElements; i += 8) {
        vec_a = _mm256_loadu_ps(&a[aIndex][i]);
        vec_b = _mm256_loadu_ps(&bT[bIndex][i]);
        sum_vec = _mm256_fmadd_ps(vec_a, vec_b, sum_vec);
    }

    float sum_array[8] = {0};
    _mm256_storeu_ps(sum_array, sum_vec);
    resultMatrix[aIndex][bIndex] = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                   sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
}


void multiplySingleResultDouble(const std::vector<std::vector<double>>& a,
                               const std::vector<std::vector<double>>& bT,
                               std::vector<std::vector<double>>& resultMatrix,
                               int aIndex, int bIndex, int numOfElements) {
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d vec_a;
    __m256d vec_b;

    for (int i = 0; i < numOfElements; i += 8) {
        vec_a = _mm256_loadu_pd(&a[aIndex][i]);
        vec_b = _mm256_loadu_pd(&bT[bIndex][i]);
        sum_vec = _mm256_fmadd_pd(vec_a, vec_b, sum_vec);
    }

    double sum_array[4] = {0};
    _mm256_storeu_pd(sum_array, sum_vec);
    resultMatrix[aIndex][bIndex] = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
}


void multiplySingleResultInt8(const std::vector<std::vector<int8_t>>& a,
                               const std::vector<std::vector<int8_t>>& bT,
                               std::vector<std::vector<int8_t>>& resultMatrix,
                               int aIndex, int bIndex, int numOfElements) {
    __m256i sum_vec = _mm256_setzero_si256();
    __m256i vec_a;
    __m256i vec_b;

    for (int i = 0; i < numOfElements; i += 8) {
        vec_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[aIndex][i]));
        vec_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bT[bIndex][i]));
        sum_vec = _mm256_add_epi8(sum_vec, _mm256_maddubs_epi16(vec_a, vec_b));
    }

    int8_t sum_array[32] = {0};
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum_array), sum_vec);
    int8_t sum = 0;
    for (int i = 0; i < 32; ++i) {
        sum += sum_array[i]; // Sum up the results
    }

    resultMatrix[aIndex][bIndex] = sum;
}


void multiplySingleResultInt(const std::vector<std::vector<int>>& a,
                             const std::vector<std::vector<int>>& bT,
                             std::vector<std::vector<int>>& resultMatrix,
                             int aIndex, int bIndex, int numOfElements) {
    __m256i sum_vec = _mm256_setzero_si256();
    __m256i vec_a;
    __m256i vec_b;

    for (int i = 0; i < numOfElements; i += 8) {
        vec_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[aIndex][i]));
        vec_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bT[bIndex][i]));
        sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(vec_a, vec_b));
    }

    int sum_array[8] = {0}; // 8 int max
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum_array), sum_vec);
    int sum = 0;
    for (int i = 0; i < 8; ++i) {
        sum += sum_array[i];
    }

    resultMatrix[aIndex][bIndex] = sum;
}


template<typename T>
void avx2_threadPoolWithBatchingAndQueueWorker(std::vector<std::vector<T>>& resultMatrix,
                                               const std::vector<std::vector<T>>& a,
                                               const std::vector<std::vector<T>>& b,
                                               int numOfElements,
                                               std::queue<std::pair<int, int>>& tasks,
                                               std::mutex& mtx, std::condition_variable& cv,
                                               bool& done,
                                               std::function<void(const std::vector<std::vector<T>>&,
                                                                  const std::vector<std::vector<T>>&,
                                                                  std::vector<std::vector<T>>&,
                                                                  int, int, int)> multiplyFunc) {
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
        multiplyFunc(a, b, resultMatrix, i, j, numOfElements);
    }
}


template <typename T>
std::vector<std::vector<T>> AVX_threadPoolWithBatchingAndQueue(const std::vector<std::vector<T>>& a,
                                                               const std::vector<std::vector<T>>& b) {
    int rowsA = a.size();
    int rowsB = b.size();
    int columnsB = b[0].size();
    int numOfElements = rowsB;

    std::vector<std::vector<T>> resultMatrix(rowsA, std::vector<T>(columnsB, 0));
    const std::vector<std::vector<T>>& newB = transposeMatrix(b); // Ensure transposeMatrix is defined

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

    // Correctly define multiplyFunc based on type
    std::function<void(const std::vector<std::vector<T>>&,
                       const std::vector<std::vector<T>>&,
                       std::vector<std::vector<T>>&,
                       int, int, int)> multiplyFunc;

    if constexpr (std::is_same_v<T, float>) {
        multiplyFunc = multiplySingleResultFloat; // Ensure this matches the signature
    } else if constexpr (std::is_same_v<T, double>) {
        multiplyFunc = multiplySingleResultDouble; // Ensure this matches the signature
    } else if constexpr (std::is_same_v<T, int8_t>) {
        multiplyFunc = multiplySingleResultInt8; // Ensure this matches the signature
    } else if constexpr (std::is_same_v<T, int>) {
        multiplyFunc = multiplySingleResultInt; // Ensure this matches the signature
    }

    // Create thread pool and capture by reference in lambda
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
