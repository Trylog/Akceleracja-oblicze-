#pragma once
#include <iostream>
#include <vector>
#include <thread>


template <typename T>
void multiplySingleColumn(std::vector<std::vector<T>>& result,
                          const std::vector<std::vector<T>>& a, const std::vector<std::vector<T>>& b,
                          int aRow, int bColumn, int numOfElements) {

    T sum = T{}; // T{} is uniform initialization. Return 0 for numeric types

    for (int k = 0; k < numOfElements; ++k) {
        sum += a[aRow][k] * b[k][bColumn];
    }

    result[aRow][bColumn] = sum;
}


namespace MatrixMultiplication {
    template <typename T>
    std::vector<std::vector<T>> singleThread(const std::vector<std::vector<T>>& a, const std::vector<std::vector<T>>& b) {
        int rowsA = a.size();
        int columnsA = a[0].size();
        int rowsB = b.size();
        int columnsB = b[0].size();

        std::vector<std::vector<T>> resultMatrix(rowsA, std::vector<T>(columnsB, 0));

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < columnsB; ++j) {
                multiplySingleColumn<T>(resultMatrix, a, b, i, j, rowsB);
            }
        }

        return resultMatrix;
    }


    template <typename T>
    std::vector<std::vector<T>> naiveMultiThreads(const std::vector<std::vector<T>>& a, const std::vector<std::vector<T>>& b) {
        int rowsA = a.size();
        int columnsA = a[0].size();
        int rowsB = b.size();
        int columnsB = b[0].size();

        std::vector<std::vector<T>> resultMatrix(rowsA, std::vector<T>(columnsB, 0));

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

        for (auto& t : threads) {
            t.join();
        }

        return resultMatrix;
    }
}