#include <gtest/gtest.h>

#include <stddef.h>
#include <stdint.h>

#include <random>

#include "bench/ethtxn.cuh"
#include "bench/wiki.cuh"
#include "bench/ycsb.cuh"
#include "mpt/cpu_mpt.cuh"
#include "mpt/gpu_mpt.cuh"
#include "mpt/node.cuh"
#include "util/timer.cuh"

struct testA {
    std::atomic<testA *> a;
    testA * b;
};

int main() {
    testA * a = new testA();
    testA * b = new testA();
    testA * c = new testA();

    a->a.store(b);
    b->b = c;
    testA * d = a->a.load();
    std::cout<<"load b: "<<d<<std::endl;
    std::cout<<"b: "<<b<<std::endl;
    std::cout<<"load b's b"<<d->b<<std::endl;
    std::cout<<"b's b: "<<b->b<<std::endl;

    testA * e = new testA();
    d->b = e;
    std::cout<<"load b's b"<<d->b<<std::endl;
    std::cout<<"b's b: "<<b->b<<std::endl;

    return 0;
}