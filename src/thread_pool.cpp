#include "include/thread_pool.h"

#include <exception>
#include <stdexcept>
#include <mutex>

namespace mini_llama {

static int kThreadCount = 0;

int GetThreadCount() {
    if (kThreadCount > 0) {
        return kThreadCount;
    }
    unsigned int count = std::thread::hardware_concurrency();
    return count > 0 ? static_cast<int>(count) : 4;
}

void SetThreadCount(int n) {
    kThreadCount = (n > 0) ? n : 0;
}

void ParallelFor(int n, const std::function<void(int begin, int end)>& work) {
    if (n <= 0) {
        return ;
    }

    int thread_count = GetThreadCount();
    constexpr int kMinChunk = 16;
    if (n < thread_count * kMinChunk) {
        work(0, n);
        return ;
    }

    if (thread_count > n) {
        thread_count = n;
    }

    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    std::exception_ptr first_exception = nullptr;
    std::mutex exception_mutex;

    int chunk = n / thread_count;
    int remaind = n % thread_count;
    int start = 0;
    for (int i = 0; i < thread_count; i++) {
        int count = chunk + (i < remaind ? 1 : 0);
        int end = start + count;
        if (count > 0) {
            threads.emplace_back(
                [start, end, &work, &first_exception, &exception_mutex]() {
                    try {
                        work(start, end);
                    } catch(...) {
                        std::lock_guard<std::mutex> lock(exception_mutex);
                        if (first_exception == nullptr) {
                            first_exception = std::current_exception();
                        }
                    }
                });
        }
        start = end;
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }

}

}