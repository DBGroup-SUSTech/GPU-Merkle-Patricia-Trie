#pragma once
#include "hash/gpu_hash_kernel.cuh"
#include "util/timer.cuh"

namespace exp_util {
class ProfileResult {};

/// @brief init env when construct, clear env when destruct
template <typename Timer>
class Profiler {
 public:
  Profiler() {}
  // virtual ~Profiler() {}

  virtual void start() { timer_.start(); }
  virtual void stop() { timer_.stop(); }
  virtual void print() { printf(" %d us\n", timer_.get()); }

 public:
  Timer timer_;
};

template <typename Timer>
class LookupProfiler : public Profiler<Timer> {
 public:
  static_assert(std::is_same_v<Timer, perf::CpuTimer<perf::us>>);

  LookupProfiler(const char* competitor, int lookup_num, int record_num)
      : competitor_(competitor),
        lookup_num_(lookup_num),
        record_num_(record_num) {}
  // virtual ~LookupProfiler() {}

  void print() override {
    printf(
        "%s lookup response time: %d us for %d operations and trie with %d "
        "records\n",
        competitor_, Profiler<Timer>::timer_.get(), lookup_num_, record_num_);
  }

 private:
  const char* competitor_;
  int lookup_num_;
  int record_num_;
};

template <typename Timer>
class InsertProfiler : public Profiler<Timer> {
 public:
  static_assert(std::is_same_v<Timer, perf::CpuTimer<perf::us>>);

  InsertProfiler(const char* competitor, int insert_num, int record_num)
      : competitor_(competitor),
        insert_num_(insert_num),
        record_num_(record_num) {}
  // virtual ~LookupProfiler() {}

  void print() override {
    int qps = insert_num_ * 1000.0 / Profiler<Timer>::timer_.get() * 1000.0;
    printf(
        "%s Insert throughput: %d qps for %d operations and trie with %d "
        "records\n",
        competitor_, qps, insert_num_, record_num_);
  }

 private:
  const char* competitor_;
  int insert_num_;
  int record_num_;
};

template <typename Timer>
class HashProfiler : public Profiler<Timer> {
 public:
  static_assert(std::is_same_v<Timer, perf::CpuTimer<perf::us>>);
  HashProfiler(const char* competitor, int insert_num, int record_num)
      : competitor_(competitor),
        insert_num_(insert_num),
        record_num_(record_num) {}
  // virtual ~LookupProfiler() {}

  void print() override {
    int qps = insert_num_ * 1000.0 / Profiler<Timer>::timer_.get() * 1000.0;
    printf(
        "%s Hash throughput: %d qps for %d operations and trie with %d "
        "records\n",
        competitor_, qps, insert_num_, record_num_);
  }

 private:
  const char* competitor_;
  int insert_num_;
  int record_num_;
};

}  // namespace exp_util