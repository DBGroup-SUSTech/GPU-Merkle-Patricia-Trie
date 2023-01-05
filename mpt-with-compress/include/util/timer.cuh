#pragma once
#include <chrono>
#include <vector>

#include "util/utils.cuh"

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

template <typename T> class CpuMultiTimer : Timer {
public:
  void start() final;
  void stop() final;
  // get() iterates each section of timeline
  int get() final;
  int get(int index) const;
  int get_longest() const;
  std::vector<clk::time_point> get_all() const;

private:
  std::vector<clk::time_point> time_points_{};
  int i{0};
};

template <typename T> void CpuMultiTimer<T>::start() {
  time_points_.reserve(5);
  time_points_.emplace_back(clk::now());
}
template <typename T> void CpuMultiTimer<T>::stop() {
  time_points_.emplace_back(clk::now());
}
template <typename T> int CpuMultiTimer<T>::get() {
  i++;
  return std::chrono::duration_cast<T>(time_points_.at(i) -
                                       time_points_.at(i - 1))
      .count();
}
template <typename T> int CpuMultiTimer<T>::get(int index) const {
  return std::chrono::duration_cast<T>(time_points_.at(index + 1) -
                                       time_points_.at(index))
      .count();
}
template <typename T> int CpuMultiTimer<T>::get_longest() const {
  return std::chrono::duration_cast<T>(
             time_points_.at(time_points_.size() - 1) - time_points_.at(0))
      .count();
}
template <typename T>
std::vector<clk::time_point> CpuMultiTimer<T>::get_all() const {
  return time_points_;
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