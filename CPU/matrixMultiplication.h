#pragma once
#include <matrixMultiplication.h>

#include <iostream> // basic I/O
#include <iomanip> // for std::setprecision

#include <vector>

// Multi-threading libraries
#include <thread>
#include <mutex>
#include<queue>
#include<condition_variable>

#include <immintrin.h> // AVX


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
void threadPoolWithBatchingAndQueueWorker(std::vector<std::vector<T>>& resultMatrix,
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


template <typename T>
void threadPoolWithBatchingWorker(std::vector<std::vector<T>>& resultMatrix,
                                  const std::vector<std::vector<T>>& a,
                                  const std::vector<std::vector<T>>& b,
                                  bool withTransposition,
                                  int numOfElements, int aMaxSize, int bMaxSize,
                                  int& aIndex, int& bIndex, std::mutex& mtx) {
    int localAIndex, localBIndex;
    while (true) {
        {
            std::lock_guard lock(mtx);

            if (bIndex == bMaxSize - 1) {
                bIndex = 0;
                aIndex++;
            } else {
                bIndex++;
            }

            if (aIndex >= aMaxSize) {
                return;
            }

            localAIndex = aIndex;
            localBIndex = bIndex;
        }

        multiplySingleColumn(resultMatrix, a, b,
                             localAIndex, localBIndex, numOfElements, withTransposition);
    }
}


std::vector<std::vector<float>> multiplyMatricesFloat(const std::vector<std::vector<float>>& a,
                                                      const std::vector<std::vector<float>>& bT) {
    int rowsA = a.size();
    int columnsA = a[0].size();
    int rowsBT = bT.size();
    std::vector<std::vector<float>> result(rowsA, std::vector<float>(rowsBT, 0));

    // Assuming b is transposed
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < rowsBT; ++j) {
            __m256 sum_vec = _mm256_setzero_ps();

            for (int k = 0; k < columnsA; k += 8) {
                __m256 vec_a = _mm256_loadu_ps(&a[i][k]);
                __m256 vec_b = _mm256_loadu_ps(&bT[j][k]);
                sum_vec = _mm256_fmadd_ps(vec_a, vec_b, sum_vec);
            }

            // Store results
            float sum_array[8] = {0};
            _mm256_storeu_ps(sum_array, sum_vec);
            result[i][j] = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                           sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
        }
    }
    return result;
}


std::vector<std::vector<double>> multiplyMatricesDouble(const std::vector<std::vector<double>>& a,
                                                        const std::vector<std::vector<double>>& bT) {
    int rowsA = a.size();
    int columnsA = a[0].size();
    int rowsBT = bT.size();
    std::vector<std::vector<double>> result(rowsA, std::vector<double>(rowsBT, 0));

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < rowsBT; ++j) {
            __m256d sum_vec = _mm256_setzero_pd();

            for (int k = 0; k < columnsA; k += 4) {
                __m256d vec_a = _mm256_loadu_pd(&a[i][k]);
                __m256d vec_b = _mm256_loadu_pd(&bT[j][k]);
                sum_vec = _mm256_fmadd_pd(vec_a, vec_b, sum_vec);
            }

            // Store results
            double sum_array[4] = {0};
            _mm256_storeu_pd(sum_array, sum_vec);
            result[i][j] = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        }
    }
    return result;
}


std::vector<std::vector<int8_t>> multiplyMatricesInt8(const std::vector<std::vector<int8_t>>& a,
                                                      const std::vector<std::vector<int8_t>>& bT) {
    int rowsA = a.size();
    int columnsA = a[0].size();
    int rowsBT = bT.size();
    std::vector<std::vector<int8_t>> result(rowsA, std::vector<int8_t>(rowsBT, 0));

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < rowsBT; ++j) {
            __m256i sum_vec = _mm256_setzero_si256();

            for (int k = 0; k < columnsA; k += 32) { // Load 32 bytes (32 int8_t)
                __m256i vec_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i][k]));
                __m256i vec_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bT[j][k]));
                sum_vec = _mm256_add_epi8(sum_vec, _mm256_maddubs_epi16(vec_a, vec_b));
            }

            // Store results
            int8_t sum_array[32] = {0}; // 32 int8_t max
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum_array), sum_vec);
            for (int m = 0; m < 32; ++m) {
                result[i][j] += sum_array[m]; // Sum up the results
            }
        }
    }
    return result;
}


std::vector<std::vector<int>> multiplyMatricesInt(const std::vector<std::vector<int>>& a,
                                                  const std::vector<std::vector<int>>& bT) {
    int rowsA = a.size();
    int columnsA = a[0].size();
    int rowsBT = bT.size();
    std::vector<std::vector<int>> result(rowsA, std::vector<int>(rowsBT, 0));

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < rowsBT; ++j) {
            __m256i sum_vec = _mm256_setzero_si256();

            for (int k = 0; k < columnsA; k += 8) {
                __m256i vec_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i][k]));
                __m256i vec_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bT[j][k]));
                sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(vec_a, vec_b));
            }

            // Store results
            int sum_array[8] = {0}; // 8 int max
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum_array), sum_vec);
            for (int m = 0; m < 8; ++m) {
                result[i][j] += sum_array[m]; // Sum up the results
            }
        }
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
    std::vector<std::vector<T>> threadPoolWithBatchingAndQueue(const std::vector<std::vector<T>>& a,
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
            threads.emplace_back(threadPoolWithBatchingAndQueueWorker<T>,
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


    template<typename T>
    std::vector<std::vector<T>> threadPoolWithBathing(const std::vector<std::vector<T>>& a,
                                                      const std::vector<std::vector<T>>& b,
                                                      bool withTransposition) {
        int rowsA = a.size();
        int columnsB = b[0].size();
        int numOfElements = b.size();

        std::vector<std::vector<T>> resultMatrix(rowsA, std::vector<T>(columnsB, 0));
        // Optionally transpose matrix b if needed
        const std::vector<std::vector<T>>& newB = withTransposition ? transposeMatrix(b) : b;

        const unsigned int maxNumberOfCPUCores = std::thread::hardware_concurrency();
        int aIndex = 0;
        int bIndex = -1;
        std::mutex mtx;

        std::vector<std::thread> threads;
        for (unsigned int n = 0; n < maxNumberOfCPUCores; ++n) {
            threads.emplace_back(threadPoolWithBatchingWorker<T>,
                                 std::ref(resultMatrix), std::cref(a), std::cref(newB), withTransposition,
                                 numOfElements, rowsA, columnsB, std::ref(aIndex), std::ref(bIndex),
                                 std::ref(mtx)
            );
        }

        for (auto& thread : threads) {
            thread.join();
        }

        return resultMatrix;
    }


    template <typename T>
    std::vector<std::vector<T>> AVX_singleThread(const std::vector<std::vector<T>>& a,
                                                 const std::vector<std::vector<T>>& b) {
        const std::vector<std::vector<T>>& newB = transposeMatrix(b);

        // Check the type of T and call the appropriate function
        if constexpr (std::is_same_v<T, float>) {
            return multiplyMatricesFloat(a, newB);
        } else if constexpr (std::is_same_v<T, double>) {
            return multiplyMatricesDouble(a, newB);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return multiplyMatricesInt8(a, newB);
        } else if constexpr (std::is_same_v<T, int>) {
            return multiplyMatricesInt(a, newB);
        }
    }
}
