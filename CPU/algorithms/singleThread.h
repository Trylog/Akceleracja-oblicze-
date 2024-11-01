#pragma once

#include <vector>

#include "avxAlignedVector.h"
#include "algorithms/_helperFunctions.h"

template<typename T>
AvxAlignedMatrix<T> singleThread(const AvxAlignedMatrix<T> &a,
                                 const AvxAlignedMatrix<T> &b,
                                 bool withTransposition) {
    int rowsA = a.size();
    int rowsB = b.size();
    int columnsB = b[0].size();
    int numOfElements = rowsB;

    AvxAlignedMatrix<T> resultMatrix = createAvxAlignedMatrix<T>(rowsA, columnsB);
    const AvxAlignedMatrix<T> &newB = withTransposition ? transposeMatrix(b) : b;

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