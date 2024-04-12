#!/bin/bash -e

run_n_times(){
    number=$1
    shift
    for n in $(seq $number); do
      $@
      sleep 2
    done
}

OUTPUT_FILE=zipf.log

BUILD_PATH=./build
GO_ETH_PATH=./go-ethereum/trie
REPEAT=5
GO_REPEAT=5

rm $OUTPUT_FILE
rm ./data/zipf.csv

# TODO: geth's code can only modified 
l="0 6 9 12 15"
for n in $l; do
    echo test...$n...
    export GMPT_ZIPF=$n
    $BUILD_PATH/utils "--gtest_filter=Util.args" "--gtest_also_run_disabled_tests"
    run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.zipfYCSB" >> $OUTPUT_FILE
    pushd $GO_ETH_PATH
      go test -run ^TestInsertYCSBzip$ -count=$GO_REPEAT -timeout 0 >> ../../$OUTPUT_FILE
    # for i in $(seq 10); do
    #   CASE_ETHT_DATA_VOLUME=$n go test -run ^TestGMPTTransactionProcessing$ -count=1 -timeout 0 >> ../../$OUTPUT_FILE
    #   sleep 2
    # done
    popd
done
