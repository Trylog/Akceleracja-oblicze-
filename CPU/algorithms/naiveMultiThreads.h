#pragma once

#include <vector>
#include <thread>

#include "avxAlignedVector.h"
#include "algorithms/_helperFunctions.h"

template<typename T>
AvxAlignedMatrix<T> naiveMultiThreads(const AvxAlignedMatrix<T> &a,
                                      const AvxAlignedMatrix<T> &b,
                                      bool withTransposition) {
    int rowsA = a.size();
    int rowsB = b.size();
    int columnsB = b[0].size();
    int numOfElements = rowsB;

    AvxAlignedMatrix<T> resultMatrix = createAvxAlignedMatrix<T>(rowsA, columnsB);
    const AvxAlignedMatrix<T> &newB = withTransposition ? transposeMatrix(b) : b;

    std::vector<std::thread> threads;

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < columnsB; ++j) {
            threads.push_back(
                    std::thread(multiplySingleColumn<T>,
                                std::ref(resultMatrix), std::cref(a), std::cref(newB),
                                i, j, numOfElements,
                                withTransposition
                    )
            );
        }
    }

    for (auto &t: threads) {
        t.join();
    }

    return resultMatrix;
}