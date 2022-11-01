#pragma once

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

} // namespace perf