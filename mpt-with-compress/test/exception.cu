#include <gtest/gtest.h>

#include <util/timer.cuh>
#include <util/utils.cuh>

#include "bench/ycsb.cuh"

TEST(Exception, PCIe) {
  // using namespace bench::ycsb;
  int size = INT32_MAX;
  char *data = new char[size];
  for (int i = 0; i < 10; ++i) {
    CHECK_ERROR(gutil::PinHost(data, size));
    char *d_data = nullptr;
    CHECK_ERROR(gutil::DeviceAlloc(d_data, size));
    perf::GpuTimer<perf::us> timer;
    timer.start();
    CHECK_ERROR(gutil::CpyHostToDevice(d_data, data, size));
    timer.stop();
    printf("transfer time: %d\n", timer.get());
    CHECK_ERROR(cudaDeviceReset());
  }
}
