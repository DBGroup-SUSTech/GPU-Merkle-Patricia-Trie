#!/bin/bash -e

run_n_times(){
    number=$1
    shift
    for n in $(seq $number); do
      $@
      sleep 2
    done
}

export GMPT_RW_RRATIO=1

n1=$GMPT_RW_RRATIO

# modify the build path
BUILD_PATH=./build
GO_ETH_PATH=./go-ethereum/trie
REPEAT=5
GO_REPEAT=5

rm ./data/rw.csv
rm test_rw.log

for n in $(seq 9); do
    $BUILD_PATH/utils "--gtest_filter=Util.args" "--gtest_also_run_disabled_tests"

    echo test_rw...
    run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.RW" >> test_rw.log
    pushd $GO_ETH_PATH
    go test -run ^TestRW$ -count=$GO_REPEAT >> ../../test_rw.log
    popd

    n1=`expr $n1 + 1`
    echo $n1
    export GMPT_RW_RRATIO=$n1
done
