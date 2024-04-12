#!/bin/bash -e
run_n_times(){
    number=$1
    shift
    for n in $(seq $number); do
      $@
      sleep 2
    done
}

# rm -f test_gpu.log
# rm -f test_cpu.log
# rm ./data/atomic_contension.csv
# rm ./data/atomic_uncontension.csv

# export GMPT_THREAD_NUM=4096
# export GMPT_LOOP_COUNT=400000000
# export GMPT_MODE=0
# export GMPT_ATOMIC_TYPE=0
export RUN_TIMES=5

# for n in $(seq 10); do
#     echo $GMPT_THREAD_NUM
#     run_n_times $RUN_TIMES ./build/thread "--gtest_filter=Atomics.AtomicGPU" >> test_gpu.log
#     run_n_times $RUN_TIMES ./build/thread "--gtest_filter=Atomics.AtomicCPU" >> test_cpu.log
#     export GMPT_THREAD_NUM=`expr $GMPT_THREAD_NUM / 2`
# done

# export GMPT_THREAD_NUM=4096
# export GMPT_LOOP_COUNT=400000000
# export GMPT_MODE=0
# export GMPT_ATOMIC_TYPE=1

# for n in $(seq 10); do
#     echo $GMPT_THREAD_NUM
#     run_n_times $RUN_TIMES ./build/thread "--gtest_filter=Atomics.AtomicGPU" >> test_gpu.log
#     run_n_times $RUN_TIMES ./build/thread "--gtest_filter=Atomics.AtomicCPU" >> test_cpu.log
#     export GMPT_THREAD_NUM=`expr $GMPT_THREAD_NUM / 2`
# done

# export GMPT_THREAD_NUM=4096
# export GMPT_LOOP_COUNT=400000000
# export GMPT_MODE=1
# export GMPT_ATOMIC_TYPE=1

# for n in $(seq 10); do
#     echo $GMPT_THREAD_NUM
#     run_n_times $RUN_TIMES ./build/thread "--gtest_filter=Atomics.AtomicGPU" >> test_gpu.log
#     run_n_times $RUN_TIMES ./build/thread "--gtest_filter=Atomics.AtomicCPU" >> test_cpu.log
#     export GMPT_THREAD_NUM=`expr $GMPT_THREAD_NUM / 2`
# done

export GMPT_THREAD_NUM=256
export GMPT_LOOP_COUNT=150000000
export GMPT_MODE=1
export GMPT_ATOMIC_TYPE=0

for n in $(seq 6); do
    echo $GMPT_THREAD_NUM
    run_n_times $RUN_TIMES ./build/thread "--gtest_filter=Atomics.AtomicGPU" >> test_gpu.log
    run_n_times $RUN_TIMES ./build/thread "--gtest_filter=Atomics.AtomicCPU" >> test_cpu.log
    export GMPT_THREAD_NUM=`expr $GMPT_THREAD_NUM / 2`
done