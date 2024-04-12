#include "util/threadpool.cuh"
#include <unistd.h>
#include <gtest/gtest.h>
#include "util/experiments.cuh"

uint64_t getTimePointNanoseconds()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto now_nanos = time_point_cast<nanoseconds>(now);
    auto value = now_nanos.time_since_epoch();
    return value.count();
}

void set_thread_with_numa(int thread_num, int numa_node) {
  auto core_ids = cutil::getCoresInNumaNode(numa_node);
  if (thread_num > core_ids.size()) {
    std::cout << "thread_num is larger than core_ids.size()" << std::endl; 
    thread_num = core_ids.size();
  }
  cutil::bind_core(core_ids, thread_num);
}

TEST(Atomics, AtomicCPU) {
    set_thread_with_numa(32, 1);
    int run_for_seconds = 5;
    std::atomic<bool> keep_running = true;
    std::atomic<uint64_t> running_threads_counter = 0;
    int num_worker = arg_util::get_record_num(arg_util::Dataset::THREADNUM);
    cpool::Mode mode = (cpool::Mode)arg_util::get_record_num(arg_util::Dataset::MODE);
    int atomic_type = arg_util::get_record_num(arg_util::Dataset::ATOMIC_TYPE);
    // int num_worker = 32;
    // cpool::Mode mode = (cpool::Mode)0;
    // int atomic_type = CAS_TYPE;
    auto cworkerPool = std::make_unique<cpool::WorkerPool>(num_worker, mode);

    for (uint64_t t_i = 0; t_i < num_worker; ++t_i) {
        (*cworkerPool).scheduleJobAsync(t_i, [&, t_i]() {
        running_threads_counter++;
        uint64_t increment = 1;
        while (keep_running) {
            // add atomic operations
            std::atomic<int> * thread_region = cpool::Worker::my().memory_region;
            if (atomic_type == CAS_TYPE) {
                // CAS 0 or 1
                int expected = rand()%2;
                int desired = 1 - expected;
                thread_region->compare_exchange_weak(expected, desired);
            } else {
                // Fetch Add
                thread_region->fetch_add(increment);
            }
            cpool::Worker::my().tx_counter++;
        }
        running_threads_counter--;
        });
    }
    // -------------------------------------------------------------------------------------
    // Join Threads
    // -------------------------------------------------------------------------------------
    sleep(run_for_seconds);
    keep_running = false;
    while (running_threads_counter) {
        _mm_pause();
    }
    (*cworkerPool).joinAll();
    
    // sum tx_counters
    long long total_tx = 0;
    for (int i=0;i<num_worker;i++) {
        total_tx += (long long)(*cworkerPool).getWorker(i)->tx_counter;
    }
    total_tx = total_tx / run_for_seconds;
    std::cout << "Throughput: " << total_tx << std::endl;

    // -------------------------------------------------------------------------------------
    std::vector<std::string> columns = {"method", "thread_num","throughput"};
    exp_util::CSVDataRecorder contension_recorder(columns, "./data/atomic_contension.csv");
    exp_util::CSVDataRecorder uncontension_recorder(columns, "./data/atomic_uncontension.csv");

    if (atomic_type == CAS_TYPE) {
        if (mode == cpool::Mode::CONTENSION) {
            contension_recorder.update_row({"CAS", std::to_string(num_worker), std::to_string(total_tx)});
        } else {
            uncontension_recorder.update_row({"CAS", std::to_string(num_worker), std::to_string(total_tx)});
        }
    } else {
        if (mode == cpool::Mode::CONTENSION) {
            contension_recorder.update_row({"FETCH_ADD", std::to_string(num_worker), std::to_string(total_tx)});
        } else {
            uncontension_recorder.update_row({"FETCH_ADD", std::to_string(num_worker), std::to_string(total_tx)});
        }
    }
    // contension_recorder.persist_data();
    // uncontension_recorder.persist_data();
}

TEST(Atomics, AtomicLoad) {
    set_thread_with_numa(32, 1);
    int run_for_seconds = 5;
    std::atomic<bool> keep_running = true;
    std::atomic<uint64_t> running_threads_counter = 0;
    int num_worker = 32;
    cpool::Mode mode = cpool::Mode::UNCONTENSION;
    int atomic_type = LOAD;
    auto cworkerPool = std::make_unique<cpool::WorkerPool>(num_worker, mode);

    for (uint64_t t_i = 0; t_i < num_worker; ++t_i) {
        (*cworkerPool).scheduleJobAsync(t_i, [&, t_i]() {
        int t = 0;
        running_threads_counter++;
        uint64_t increment = 1;
        while (keep_running) {
            // add atomic operations
            std::atomic<int> * thread_region = cpool::Worker::my().memory_region;
            int * no_atomic_thread_region = cpool::Worker::my().no_atomic_memory_region;
            if (atomic_type == LOAD) {
                // CAS 0 or 1
                t = thread_region->load(std::memory_order_acquire);
            } else if (atomic_type == RELAX_LOAD) {
                // Fetch Add
                t = thread_region->load(std::memory_order_relaxed);
            } else {
                t = *no_atomic_thread_region;
            }
            cpool::Worker::my().tx_counter++;
        }
        std::cout << "t: " << t << std::endl;
        running_threads_counter--;
        });
    }
    // -------------------------------------------------------------------------------------
    // Join Threads
    // -------------------------------------------------------------------------------------
    sleep(run_for_seconds);
    keep_running = false;
    while (running_threads_counter) {
        _mm_pause();
    }
    (*cworkerPool).joinAll();
    // sum tx_counters
    long long total_tx = 0;
    for (int i=0;i<num_worker;i++) {
        total_tx += (long long)(*cworkerPool).getWorker(i)->tx_counter;
    }
    total_tx = total_tx / run_for_seconds;
    std::cout << "Throughput: " << total_tx << std::endl; 
}

TEST(Atomics, AtomicGPU) {
    int run_for_seconds = 10;
    int num_worker = arg_util::get_record_num(arg_util::Dataset::THREADNUM);
    gpool::Mode mode = (gpool::Mode)arg_util::get_record_num(arg_util::Dataset::MODE);
    int atomic_type = arg_util::get_record_num(arg_util::Dataset::ATOMIC_TYPE);
    int loop_count = arg_util::get_record_num(arg_util::Dataset::LOOP_COUNT);
    // int num_worker = 32;
    // gpool::Mode mode = (gpool::Mode)0;
    // int atomic_type = CAS_TYPE;

    std::vector<std::string> columns = {"method", "thread_num","throughput"};
    exp_util::CSVDataRecorder contension_recorder(columns, "./data/atomic_contension.csv");
    exp_util::CSVDataRecorder uncontension_recorder(columns, "./data/atomic_uncontension.csv"); 
    long long total_tx = gpool::run_pool(num_worker, mode, atomic_type, loop_count);
    std::cout << "Throughput: " << total_tx << std::endl;

    if (atomic_type == CAS_TYPE) {
        if (mode == gpool::Mode::CONTENSION) {
            contension_recorder.update_row({"GCAS", std::to_string(num_worker), std::to_string(total_tx)});
        } else {
            uncontension_recorder.update_row({"GCAS", std::to_string(num_worker), std::to_string(total_tx)});
        }
    } else {
        if (mode == gpool::Mode::CONTENSION) {
            contension_recorder.update_row({"GFETCH_ADD", std::to_string(num_worker), std::to_string(total_tx)});
        } else {
            uncontension_recorder.update_row({"GFETCH_ADD", std::to_string(num_worker), std::to_string(total_tx)});
        }
    }
    contension_recorder.persist_data();
    uncontension_recorder.persist_data();
}

TEST (BASIC, bignumber) {
    long long a = 4096;
    long long c = 400000000;

    std::cout << a * c << std::endl;

    std::cout << (a * c) / 16290156 * 1000 *1000  << std::endl;
}