#pragma once

#include <vector>

#include "_helperFunctions.h"

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