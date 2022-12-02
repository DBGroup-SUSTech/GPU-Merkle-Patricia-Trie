#pragma once
#include "util/utils.cuh"
#include <chrono>

namespace perf {
using clk = std::chrono::system_clock;
using ms = std::chrono::milliseconds;
using us = std::chrono::microseconds;
using ns = std::chrono::nanoseconds;
using s = std::chrono::seconds;

class Timer {
public:
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual int get() = 0;
};

template <typename T> class CpuTimer : Timer {
public:
  void start() final;
  void stop() final;
  int get() final;

private:
  clk::time_point before_;
  clk::time_point after_;
};

template <typename T> void CpuTimer<T>::start() { before_ = clk::now(); }
template <typename T> void CpuTimer<T>::stop() { after_ = clk::now(); }
template <typename T> int CpuTimer<T>::get() {
  return std::chrono::duration_cast<T>(after_ - before_).count();
}

template <typename T> class GpuTimer : Timer {
  static_assert(std::is_same_v<T, ms>);

public:
  GpuTimer(cudaStream_t stream = (cudaStream_t)0) : stream_(stream) {
    CHECK_ERROR(cudaEventCreate(&start_));
    CHECK_ERROR(cudaEventCreate(&stop_));
  }

public:
  void start() final;
  void stop() final;
  int get() final;

private:
  cudaStream_t stream_;
  cudaEvent_t start_;
  cudaEvent_t stop_;
};
template <typename T> void GpuTimer<T>::start() {
  CHECK_ERROR(cudaEventRecord(start_, stream_));
}
template <typename T> void GpuTimer<T>::stop() {
  CHECK_ERROR(cudaEventRecord(stop_, stream_));
}
template <typename T> int GpuTimer<T>::get() {
  float ms;
  CHECK_ERROR(cudaEventElapsedTime(&ms, start_, stop_));
  return static_cast<int>(ms);
}
} // namespace perf