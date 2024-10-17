#include "matrixOperations.h"

#include <iostream>
#include <vector>
#include <thread>


void multiplySingleColumn(std::vector<std::vector<int>>& result,
        const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b,
        int aRow, int bColumn, int numOfElements) {
    
    int sum = 0;

    for (int k = 0; k < numOfElements; ++k) {
        sum += a[aRow][k] * b[k][bColumn];
    }

    result[aRow][bColumn] = sum;
}

namespace MatrixOps {
    std::vector<std::vector<int>> multiply(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b) {
        int rowsA = a.size();
        int columnsA = a[0].size();
        int rowsB = b.size();
        int columnsB = b[0].size();

        if (columnsA != rowsB) {
            throw std::invalid_argument("Number of columns in A must be equal to number of rows in B");
        }

        std::vector<std::vector<int>> result(rowsA, std::vector<int>(columnsB, 0));

        std::vector<std::thread> threads;

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < columnsB; ++j) {
                threads.push_back(
                    std::thread(multiplySingleColumn, std::ref(result), a, b, i, j, columnsA)
                );
            }
        }

        for (auto& t : threads) {
            t.join();
        }

        return result;
    }

    std::vector<std::vector<int>> multiplyOnSingleThread(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b) {
        int rowsA = a.size();
        int columnsA = a[0].size();
        int rowsB = b.size();
        int columnsB = b[0].size();

        std::vector<std::vector<int>> result(rowsA, std::vector<int>(columnsB, 0));

        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < columnsB; ++j) {
                multiplySingleColumn(result, a, b, i, j, rowsB);
            }
        }

        return result;
    }
}