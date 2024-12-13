#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "avxAlignedVector.h"
#include "algorithms/_helperFunctions.h"

template <typename T>
void threadPoolWithBatchingWorker(AvxAlignedMatrix<T> &resultMatrix,
                                  const AvxAlignedMatrix<T> &a,
                                  const AvxAlignedMatrix<T> &b,
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


template<typename T>
AvxAlignedMatrix<T> threadPoolWithBathing(const AvxAlignedMatrix<T>& a,
                                          const AvxAlignedMatrix<T>& b,
                                          bool withTransposition) {
    int rowsA = a.size();
    int columnsB = b[0].size();
    int numOfElements = b.size();

    AvxAlignedMatrix<T> resultMatrix = createAvxAlignedMatrix<T>(rowsA, columnsB);

    const AvxAlignedMatrix<T> &newB = withTransposition ? transposeMatrix(b) : b;

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