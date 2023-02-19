#!/bin/bash -e

run_n_times(){
    number=$1
    shift
    for n in $(seq $number); do
      $@
      sleep 2
    done
}

export GMPT_TRIESIZE=320000
n1=$GMPT_TRIESIZE

BUILD_PATH=./build
GO_ETH_PATH=./go-ethereum/trie
REPEAT=10

rm test_sup_experiments.log

for n in $(seq 7); do
    $BUILD_PATH/utils "--gtest_filter=Util.args" "--gtest_also_run_disabled_tests"

    echo test_trie_size...
    run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.TrieSizeEthtxn" >> test_sup_experiments.log
    pushd $GO_ETH_PATH
    go test -run ^TestEthTrieSize$ -count=$REPEAT >> ../../test_sup_experiments.log
    popd
    
    n1=`expr $n1 + 40000 `
    echo $n1
    export GMPT_TRIESIZE=$n1
done

echo test_async...
run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.AsyncMemcpyEthtxn" >> test_sup_experiments.log
