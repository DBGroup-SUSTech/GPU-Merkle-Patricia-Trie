#include "utils.cuh"
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <curand_kernel.h>
#include <unistd.h>
#include "util/timer.cuh"

namespace cpool {

enum class Mode {
   UNCONTENSION,
   CONTENSION
};

struct Worker{
   static thread_local Worker* tlsPtr;
   static inline Worker& my() { return *Worker::tlsPtr; }
   std::atomic<int> * memory_region = nullptr;
   int * no_atomic_memory_region = nullptr;
   int tx_counter = 0;
   int latency_counter = 0;

   Worker(std::atomic<int> * mr, int* no_a_mr):memory_region(mr), no_atomic_memory_region(no_a_mr) {}

   void reset(std::atomic<int> * mr) {
      memory_region = mr;
      tx_counter = 0;
      latency_counter = 0;
   }
};

thread_local Worker* Worker::tlsPtr = nullptr;

class WorkerPool
{
   static constexpr uint64_t MAX_WORKER_THREADS = 4100;

   std::atomic<uint64_t> runningThreads = 0;
   std::atomic<bool> keepRunning = true;
   // -------------------------------------------------------------------------------------
   struct WorkerThread {
      std::mutex mutex;
      std::condition_variable cv;
      std::function<void()> job;
      bool wtReady = true;
      bool jobSet = false;
      bool jobDone = false;
   };
   // -------------------------------------------------------------------------------------
   std::vector<std::thread> workerThreads;
   std::vector<Worker*> workers;
   WorkerThread workerThreadsMeta [MAX_WORKER_THREADS];
   uint32_t workersCount;
   std::atomic<int> memory_regions [MAX_WORKER_THREADS];
   int no_atomic_memory_regions [MAX_WORKER_THREADS];
  public:
   
   // -------------------------------------------------------------------------------------
   WorkerPool(int num_w, Mode m);
   ~WorkerPool();
   // -------------------------------------------------------------------------------------
   void scheduleJobAsync(uint64_t t_i, std::function<void()> job);
   void scheduleJobSync(uint64_t t_i, std::function<void()> job);
   void joinAll();

   Worker * getWorker(uint64_t t_i) {
      return workers[t_i];
   }
};

// WorkerPool::WorkerPool(): workers(MAX_WORKER_THREADS,nullptr)
WorkerPool::WorkerPool(int num_w, Mode m)
{
   workersCount = num_w;
   assert(workersCount < MAX_WORKER_THREADS);
   workerThreads.reserve(workersCount);
   workers.reserve(workersCount);
   for (uint64_t t_i = 0; t_i < workersCount; t_i++) {
      workerThreads.emplace_back([&, t_i]() {
         std::string threadName("worker_" + std::to_string(t_i));
         pthread_setname_np(pthread_self(), threadName.c_str());
         // -------------------------------------------------------------------------------------
         memory_regions[t_i].store(0);
         no_atomic_memory_regions[t_i] = 0;
         std::atomic<int> * my_region;
         int * my_no_atomic_memory_region;
         if (m == Mode::CONTENSION) {
            my_region = &memory_regions[0];
            my_no_atomic_memory_region = &no_atomic_memory_regions[t_i];
         } else if (m == Mode::UNCONTENSION) {
            my_region = &memory_regions[t_i];
            my_no_atomic_memory_region = &no_atomic_memory_regions[0];
         } else {
            assert(false);
         }
         workers[t_i] = new Worker(my_region, my_no_atomic_memory_region);
         Worker::tlsPtr = workers[t_i];
         // -------------------------------------------------------------------------------------
         runningThreads++;
         auto& meta = workerThreadsMeta[t_i];
         while (keepRunning) {
            std::unique_lock guard(meta.mutex);
            meta.cv.wait(guard, [&]() { return keepRunning == false || meta.jobSet; });
            if (!keepRunning) {
               break;
            }
            meta.wtReady = false;
            meta.job();
            meta.wtReady = true;
            meta.jobDone = true;
            meta.jobSet = false;
            meta.cv.notify_one();
         }
         runningThreads--;
      });
   }   
   
   for (auto& t : workerThreads) {
      t.detach();
   }
   // -------------------------------------------------------------------------------------
   // Wait until all worker threads are initialized
   while (runningThreads < workersCount) {
   }
}
// -------------------------------------------------------------------------------------
WorkerPool::~WorkerPool(){
   keepRunning = false;
   
   for (uint64_t t_i = 0; t_i < workersCount; t_i++) {
      workerThreadsMeta[t_i].cv.notify_one();
   }
   while (runningThreads) {
   }
}
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
void WorkerPool::scheduleJobSync(uint64_t t_i, std::function<void()> job)
{
   assert(t_i < workersCount);
   auto& meta = workerThreadsMeta[t_i];
   std::unique_lock guard(meta.mutex);
   meta.cv.wait(guard, [&]() { return !meta.jobSet && meta.wtReady; });
   meta.jobSet = true;
   meta.jobDone = false;
   meta.job = job;
   guard.unlock();
   meta.cv.notify_one();
   guard.lock();
   meta.cv.wait(guard, [&]() { return meta.jobDone; });
}
// -------------------------------------------------------------------------------------
void WorkerPool::scheduleJobAsync(uint64_t t_i, std::function<void()> job)
{
   assert(t_i < workersCount);
   auto& meta = workerThreadsMeta[t_i];
   std::unique_lock guard(meta.mutex);
   meta.cv.wait(guard, [&]() { return !meta.jobSet && meta.wtReady; });
   meta.jobSet = true;
   meta.jobDone = false;
   meta.job = job;
   guard.unlock();
   meta.cv.notify_one();
}
// -------------------------------------------------------------------------------------
void WorkerPool::joinAll()
{
   for (uint64_t t_i = 0; t_i < workersCount; t_i++) {
      auto& meta = workerThreadsMeta[t_i];
      std::unique_lock guard(meta.mutex);
      meta.cv.wait(guard, [&]() { return meta.wtReady && !meta.jobSet; });
   }
}
}


namespace gpool {

enum class Mode {
   UNCONTENSION,
   CONTENSION
};

__global__ void random_setup(curandState *d_states, int n)
{
   int wid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
   if (wid >= n)
   {   
      return;
   }
   int lid_w = threadIdx.x % 32;
   if (lid_w > 0)
   {
      return;
   }
   curand_init(1, wid, 0, &d_states[wid]);
}

__device__ __forceinline__ void worker_thread(int * memory_region, uint32_t * tx_counter, int * latency_counter, int atomic_type, curandState state, int loop_count) {
   int i = 0;
   while (i < loop_count) {
      if(atomic_type == CAS_TYPE) {
         int expected = curand(&state) % 2;
         int desired = 1 - expected;
         atomicCAS(memory_region, expected, desired);
      } else {
         atomicAdd(memory_region, 1);
      }
      *tx_counter += 1;
      i++;
   }
}

__global__ void atomic_kernel(int * memory_regions, Mode m, int atomic_type, uint32_t * txn_counters, curandState *d_states, int n, int loop_count) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   if (tid >= n)
   {
      return;
   }
   if (m == Mode::CONTENSION) {
      worker_thread(memory_regions, &txn_counters[tid], nullptr, atomic_type, curandState(), loop_count);
   } else {
      worker_thread(&memory_regions[tid], &txn_counters[tid], nullptr, atomic_type, d_states[tid], loop_count);
   }
}

long long run_pool(int n, Mode m, int atomic_type, int loop_count) {
   int * d_memory_regions;
   CHECK_ERROR(gutil::DeviceAlloc(d_memory_regions, n));
   uint32_t * d_txn_counters;
   CHECK_ERROR(gutil::DeviceAlloc(d_txn_counters, n));

   CHECK_ERROR(gutil::DeviceSet(d_memory_regions, 0, n));
   CHECK_ERROR(gutil::DeviceSet(d_txn_counters, 0, n));

   curandState_t *d_states;
   CHECK_ERROR(gutil::DeviceAlloc(d_states,n));
   // puts
   const int block_size = 128;
   const int num_blocks = (n + block_size - 1) / block_size;

   const int rpwarp_block_size = 256;
   const int rpwarp_num_blocks = (n * 32 + rpwarp_block_size - 1) /
                                    rpwarp_block_size;

   random_setup<<<rpwarp_num_blocks, rpwarp_block_size>>>(d_states,n);
   
   CHECK_ERROR(cudaDeviceSynchronize());
   perf::GpuTimer<perf::us> timer;
   timer.start();

   atomic_kernel<<<num_blocks, block_size>>>(d_memory_regions, m, atomic_type, d_txn_counters, d_states, n, loop_count);
   CHECK_ERROR(cudaDeviceSynchronize());
   timer.stop();

   int time = timer.get();
   // uint32_t * h_txn_counters = new uint32_t[n];
   // CHECK_ERROR(gutil::CpyDeviceToHost(h_txn_counters, d_txn_counters, n));
   
   // int total_txns = 0;
   // for (int i = 0; i < n; i++) {
   //    printf("thread %d: %d\n", i, h_txn_counters[i]);
   //    total_txns += h_txn_counters[i];
   // }
   // printf("total txns: %d\n", total_txns);
   std::cout << time << "us" << std::endl;

   long long total_txns = (long long)n * (long long)loop_count;
   long long throughput = (total_txns / time) * 1000*1000;

   std::cout << total_txns << "tx" <<std::endl;
   std::cout<< throughput << "tp" << std::endl;

   return throughput;
}
}