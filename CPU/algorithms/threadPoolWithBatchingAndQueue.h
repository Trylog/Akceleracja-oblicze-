#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "algorithms/_helperFunctions.h"

template<typename T>
void threadPoolWithBatchingAndQueueWorker(std::vector<std::vector<T>>& resultMatrix,
                                          const std::vector<std::vector<T> >& a, const std::vector<std::vector<T>>& b, int numOfElements,
                                          std::queue<std::pair<int, int>>& tasks,
                                          std::mutex& mtx, std::condition_variable& cv,
                                          bool& done) {
    while (true) {
        std::pair<int, int> task;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]() { return done || !tasks.empty(); });

            if (done && tasks.empty()) return;

            task = tasks.front();
            tasks.pop();
        }

        int i = task.first;
        int j = task.second;
        multiplySingleColumn(resultMatrix, a, b,
                             i, j, numOfElements,
                             true);
    }
}


template <typename T>
std::vector<std::vector<T>> threadPoolWithBatchingAndQueue(const std::vector<std::vector<T>>& a,
                                                           const std::vector<std::vector<T>>& b,
                                                           bool withTransposition) {
    int rowsA = a.size();
    int rowsB = b.size();
    int columnsB = b[0].size();
    int numOfElements = rowsB;

    std::vector<std::vector<T>> resultMatrix(rowsA, std::vector<T>(columnsB, 0));
    const std::vector<std::vector<T>>& newB = withTransposition ? transposeMatrix(b) : b;

    const unsigned int maxNumberOfCPUCores = std::thread::hardware_concurrency();

    std::queue<std::pair<int, int>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool done = false;

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < columnsB; ++j) {
            tasks.emplace(i, j);
        }
    }

    std::vector<std::thread> threads;
    for (unsigned int n = 0; n < maxNumberOfCPUCores; ++n) {
        threads.emplace_back(threadPoolWithBatchingAndQueueWorker<T>,
                             std::ref(resultMatrix), std::cref(a), std::cref(b), numOfElements,
                             std::ref(tasks), std::ref(mtx), std::ref(cv), std::ref(done)
        );
    }

    cv.notify_all();

    {
        std::unique_lock<std::mutex> lock(mtx);
        done = true;
    }
    cv.notify_all();
    for (auto& thread : threads) {
        thread.join();
    }

    return resultMatrix;
}