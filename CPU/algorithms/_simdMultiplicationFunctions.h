#pragma once

#include <immintrin.h> // AVX
#include <vector>
#include <cstdint>

void multiplySingleResultFloat(const std::vector<std::vector<float>>& a,
                               const std::vector<std::vector<float>>& bT,
                               std::vector<std::vector<float>>& resultMatrix,
                               int aIndex, int bIndex, int numOfElements, int numOfElementsThatFitSimd) {
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 vec_a;
    __m256 vec_b;

    int i = 0;
    for (; i < numOfElementsThatFitSimd; i += 8) {
        vec_a = _mm256_loadu_ps(&a[aIndex][i]);
        vec_b = _mm256_loadu_ps(&bT[bIndex][i]);
        sum_vec = _mm256_fmadd_ps(vec_a, vec_b, sum_vec);
    }

    float sum_array[8] = {0};
    _mm256_storeu_ps(sum_array, sum_vec);
    float sum = 0;
    sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
          sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

    for (; i < numOfElements; ++i) {
        sum += a[aIndex][i] * bT[bIndex][i];
    }

    resultMatrix[aIndex][bIndex] = sum;
}


void multiplySingleResultDouble(const std::vector<std::vector<double>>& a,
                                const std::vector<std::vector<double>>& bT,
                                std::vector<std::vector<double>>& resultMatrix,
                                int aIndex, int bIndex, int numOfElements, int numOfElementsThatFitSimd) {
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d vec_a;
    __m256d vec_b;

    int i = 0;
    for (; i < numOfElementsThatFitSimd; i += 4) {
        vec_a = _mm256_loadu_pd(&a[aIndex][i]);
        vec_b = _mm256_loadu_pd(&bT[bIndex][i]);
        sum_vec = _mm256_fmadd_pd(vec_a, vec_b, sum_vec);
    }

    double sum_array[4] = {0};
    _mm256_storeu_pd(sum_array, sum_vec);
    double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    for (; i < numOfElements; ++i) {
        sum += a[aIndex][i] * bT[bIndex][i];
    }

    resultMatrix[aIndex][bIndex] = sum;
}


void multiplySingleResultInt8(const std::vector<std::vector<int8_t>>& a,
                              const std::vector<std::vector<int8_t>>& bT,
                              std::vector<std::vector<int8_t>>& resultMatrix,
                              int aIndex, int bIndex, int numOfElements, int numOfElementsThatFitSimd) {
    __m256i sum_vec = _mm256_setzero_si256();
    __m256i vec_a;
    __m256i vec_b;

    int i = 0;
    for (; i < numOfElementsThatFitSimd; i += 32) {
        vec_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[aIndex][i]));
        vec_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bT[bIndex][i]));
        sum_vec = _mm256_add_epi8(sum_vec, _mm256_maddubs_epi16(vec_a, vec_b));
    }

    int8_t sum_array[32] = {0};
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum_array), sum_vec);
    int8_t sum = 0;
    for (int j = 0; j < 32; ++j) {
        sum += sum_array[j];
    }

    for (; i < numOfElements; ++i) {
        sum += a[aIndex][i] * bT[bIndex][i];
    }

    resultMatrix[aIndex][bIndex] = sum;
}


void multiplySingleResultInt(const std::vector<std::vector<int>>& a,
                             const std::vector<std::vector<int>>& bT,
                             std::vector<std::vector<int>>& resultMatrix,
                             int aIndex, int bIndex, int numOfElements, int numOfElementsThatFitSimd) {
    __m256i sum_vec = _mm256_setzero_si256();
    __m256i vec_a;
    __m256i vec_b;

    int i = 0;
    for (; i < numOfElementsThatFitSimd; i += 8) {
        vec_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[aIndex][i]));
        vec_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bT[bIndex][i]));
        sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(vec_a, vec_b));
    }

    int sum_array[8] = {0}; // 8 int max
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum_array), sum_vec);
    int sum = 0;
    for (int j = 0; j < 8; ++j) {
        sum += sum_array[j];
    }

    for (; i < numOfElements; ++i) {
        sum += a[aIndex][i] * bT[bIndex][i];
    }

    resultMatrix[aIndex][bIndex] = sum;
}