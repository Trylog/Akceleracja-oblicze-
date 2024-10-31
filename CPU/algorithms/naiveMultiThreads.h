#pragma once

#include <vector>
#include <thread>

#include "_helperFunctions.h"

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