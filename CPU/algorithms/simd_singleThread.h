#pragma once

#include <immintrin.h> // AVX
#include <vector>
#include <cstdint>

std::vector<std::vector<float>> multiplyMatricesFloat(const std::vector<std::vector<float>>& a,
                                                      const std::vector<std::vector<float>>& bT) {
    int rowsA = a.size();
    int columnsA = a[0].size();
    int rowsBT = bT.size();
    std::vector<std::vector<float>> result(rowsA, std::vector<float>(rowsBT, 0));

    __m256 sum_vec;
    __m256 vec_a;
    __m256 vec_b;

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < rowsBT; ++j) {
            sum_vec = _mm256_setzero_ps();

            for (int k = 0; k < columnsA; k += 8) {
                vec_a = _mm256_loadu_ps(&a[i][k]);
                vec_b = _mm256_loadu_ps(&bT[j][k]);
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

    __m256d sum_vec;
    __m256d vec_a;
    __m256d vec_b;

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < rowsBT; ++j) {
            sum_vec = _mm256_setzero_pd();

            for (int k = 0; k < columnsA; k += 4) {
                vec_a = _mm256_loadu_pd(&a[i][k]);
                vec_b = _mm256_loadu_pd(&bT[j][k]);
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

    __m256i sum_vec;
    __m256i vec_a;
    __m256i vec_b;

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < rowsBT; ++j) {
            sum_vec = _mm256_setzero_si256();

            for (int k = 0; k < columnsA; k += 32) { // Load 32 bytes (32 int8_t)
                vec_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i][k]));
                vec_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bT[j][k]));
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

    __m256i sum_vec;
    __m256i vec_a;
    __m256i vec_b;

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < rowsBT; ++j) {
            sum_vec = _mm256_setzero_si256();

            for (int k = 0; k < columnsA; k += 8) {
                vec_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i][k]));
                vec_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bT[j][k]));
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

    return result;
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