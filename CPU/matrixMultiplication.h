#pragma once

#include <iostream> // basic I/O
#include <iomanip> // for std::setprecision

#include <vector>

// Multi-threading libraries
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "algorithms/singleThread.h"
#include "algorithms/simd_singleThread.h"
#include "algorithms/naiveMultiThreads.h"
#include "algorithms/threadPoolWithBatching.h"
#include "algorithms/threadPoolWithBatchingAndQueue.h"



namespace MatrixMultiplication {
    template<typename T>
    std::vector<std::vector<T> > singleThread(const std::vector<std::vector<T> > &a,
                                              const std::vector<std::vector<T> > &b,
                                              bool withTransposition) {
        return ::singleThread(a, b, withTransposition);
    }


    template<typename T>
    std::vector<std::vector<T> > naiveMultiThreads(const std::vector<std::vector<T> > &a,
                                                   const std::vector<std::vector<T> > &b,
                                                   bool withTransposition) {
        return ::naiveMultiThreads(a, b, withTransposition);
    }


    template <typename T>
    std::vector<std::vector<T>> threadPoolWithBatchingAndQueue(const std::vector<std::vector<T>>& a,
                                                               const std::vector<std::vector<T>>& b,
                                                               bool withTransposition) {
        return ::threadPoolWithBatchingAndQueue(a, b, withTransposition);
    }


    template<typename T>
    std::vector<std::vector<T>> threadPoolWithBathing(const std::vector<std::vector<T>>& a,
                                                      const std::vector<std::vector<T>>& b,
                                                      bool withTransposition) {
        return ::threadPoolWithBathing(a, b, withTransposition);
    }


    template <typename T>
    std::vector<std::vector<T>> AVX_singleThread(const std::vector<std::vector<T>>& a,
                                                 const std::vector<std::vector<T>>& b) {
        return ::AVX_singleThread(a, b);
    }
}
