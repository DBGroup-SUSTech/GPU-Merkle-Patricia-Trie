#include "mpt/cpu_mpt.h"
#include "mpt/gpu_mpt.cuh"

#include "util/util.h"
#include <stdio.h>

using value_t = char[8];
int main() {
  CpuMPT<int, int> cpu_mpt;
}
