#include "mpt/cpu_mpt.h"
#include "mpt/gpu_mpt.cuh"

#include "util/util.h"
#include <stdio.h>

using value_t = char[8];
int main() {
  value_t *values = new value_t[100];
  memset(values, 0, 100 * 8);
  values[8][4] = '1';
  printf("%c\n", ((char*)values)[8 * 8 + 4]);

  CpuMPT<int, int> cpu_mpt;
}
