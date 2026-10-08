// Persistent worker pool for the CPU backend: parallel_for over an index range with
// dynamic scheduling (an atomic chunk counter), the calling thread working alongside.
// MSPLAT_THREADS overrides the worker count (default: hardware concurrency).
#ifndef MSPLAT_CPU_PARALLEL_HPP
#define MSPLAT_CPU_PARALLEL_HPP

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace msplat_cpu {

class ThreadPool {
  public:
    static ThreadPool &instance() {
        static ThreadPool pool;
        return pool;
    }

    int size() const { return (int)workers_.size() + 1; }

    // fn(begin, end) over [0, n) in chunks of `grain`; blocks until all are done.
    // Exceptions thrown by fn are rethrown here (the first one wins).
    void parallel_for(size_t n, size_t grain, const std::function<void(size_t, size_t)> &fn) {
        if (n == 0) return;
        grain = std::max<size_t>(grain, 1);
        if (workers_.empty() || n <= grain || in_parallel()) {  // nested calls run inline
            fn(0, n);
            return;
        }
        std::unique_lock<std::mutex> job_lock(job_mutex_);  // one parallel_for at a time
        Job job;
        job.fn = &fn;
        job.n = n;
        job.grain = grain;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = &job;
            generation_++;
            active_ = (int)workers_.size();
        }
        cv_.notify_all();
        run(job);
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [&] { return active_ == 0; });
            job_ = nullptr;
        }
        if (job.error) std::rethrow_exception(job.error);
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &t : workers_) t.join();
    }

  private:
    struct Job {
        const std::function<void(size_t, size_t)> *fn = nullptr;
        size_t n = 0, grain = 1;
        std::atomic<size_t> next{0};
        std::exception_ptr error;
        std::mutex error_mutex;
    };

    ThreadPool() {
        int n = (int)std::thread::hardware_concurrency();
        if (const char *env = std::getenv("MSPLAT_THREADS")) n = std::max(1, std::atoi(env));
        n = std::max(1, n);
        for (int i = 0; i < n - 1; i++) workers_.emplace_back([this] { loop(); });
    }

    static bool &in_parallel() {
        static thread_local bool flag = false;
        return flag;
    }

    static void run(Job &job) {
        struct Flag {
            bool prev;
            Flag() : prev(in_parallel()) { in_parallel() = true; }
            ~Flag() { in_parallel() = prev; }
        } flag;
        for (;;) {
            size_t begin = job.next.fetch_add(job.grain);
            if (begin >= job.n) return;
            size_t end = std::min(job.n, begin + job.grain);
            try {
                (*job.fn)(begin, end);
            } catch (...) {
                std::lock_guard<std::mutex> lock(job.error_mutex);
                if (!job.error) job.error = std::current_exception();
                job.next.store(job.n);
            }
        }
    }

    void loop() {
        uint64_t seen = 0;
        for (;;) {
            Job *job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                job = job_;
            }
            if (job) run(*job);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--active_ == 0) done_cv_.notify_all();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_, job_mutex_;
    std::condition_variable cv_, done_cv_;
    Job *job_ = nullptr;
    uint64_t generation_ = 0;
    int active_ = 0;
    bool stop_ = false;
};

inline void parallel_for(size_t n, size_t grain, const std::function<void(size_t, size_t)> &fn) {
    ThreadPool::instance().parallel_for(n, grain, fn);
}

inline int num_threads() { return ThreadPool::instance().size(); }

}  // namespace msplat_cpu

#endif
