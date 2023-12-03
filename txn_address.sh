#!/bin/bash -e

run_n_times(){
    number=$1
    shift
    for n in $(seq $number); do
      $@
      sleep 2
    done
}

rm address1.log
CMD=/home/ymx/ccnpro/GPU-Merkle-Patricia-Trie/build/read_address 
PARAM1="--gtest_filter=ethtxn.memory_analysis" 
PARAM2="--gtest_also_run_disabled_tests"
export GMPT_TXN_NUM=5000

rm $OUTPUT_FILE
# TODO: geth's code can only modified 
l="5000 40000 160000 320000 640000"
for n in $l; do
    echo test...$n...
    export GMPT_TXN_NUM=$n
    $CMD $PARAM1 $PARAM2 >> address1.log
done
