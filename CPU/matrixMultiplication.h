#pragma once

#include <iostream> // basic I/O
#include <iomanip> // for std::setprecision

#include <vector>

#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "avxAlignedVector.h"
#include "algorithms/singleThread.h"
#include "algorithms/simd_singleThread.h"
#include "algorithms/naiveMultiThreads.h"
#include "algorithms/threadPoolWithBatching.h"
#include "algorithms/threadPoolWithBatchingAndQueue.h"
#include "algorithms/simd_threadPoolWithBatchingAndQueue.h"


namespace MatrixMultiplication {
    template<typename T>
    AvxAlignedMatrix<T> singleThread(const AvxAlignedMatrix<T> &a,
                                     const AvxAlignedMatrix<T> &b,
                                     bool withTransposition) {
        return ::singleThread(a, b, withTransposition);
    }


    template<typename T>
    AvxAlignedMatrix<T> naiveMultiThreads(const AvxAlignedMatrix<T> &a,
                                          const AvxAlignedMatrix<T> &b,
                                          bool withTransposition) {
        return ::naiveMultiThreads(a, b, withTransposition);
    }


    template<typename T>
    AvxAlignedMatrix<T> threadPoolWithBatchingAndQueue(const AvxAlignedMatrix<T> &a,
                                                       const AvxAlignedMatrix<T> &b,
                                                       bool withTransposition) {
        return ::threadPoolWithBatchingAndQueue(a, b, withTransposition);
    }


    template<typename T>
    AvxAlignedMatrix<T> threadPoolWithBatching(const AvxAlignedMatrix<T> &a,
                                              const AvxAlignedMatrix<T> &b,
                                              bool withTransposition) {
        return ::threadPoolWithBathing(a, b, withTransposition);
    }


    template<typename T>
    AvxAlignedMatrix<T> AVX_singleThread(const AvxAlignedMatrix<T> &a,
                                         const AvxAlignedMatrix<T> &b,
                                         bool withTransposition) {
        return ::AVX_singleThread(a, b, withTransposition);
    }


    template<typename T>
    AvxAlignedMatrix<T> AVX_threadPoolWithBatchingAndQueue(const AvxAlignedMatrix<T> &a,
                                                           const AvxAlignedMatrix<T> &b,
                                                           bool withTransposition) {
        return ::AVX_threadPoolWithBatchingAndQueue(a, b, withTransposition);
    }
}
