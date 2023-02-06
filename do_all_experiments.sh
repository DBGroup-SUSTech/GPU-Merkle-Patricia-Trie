#!/bin/bash -e

run_n_times(){
    number=$1
    shift
    for n in $(seq $number); do
      $@
      sleep 2
    done
}

export GMPT_WIKI_DATA_VOLUME=320000
export GMPT_ETH_DATA_VOLUME=640000
export GMPT_YCSB_DATA_VOLUME=1280000
export GMPT_DATA_LOOKUP_VOLUME=1280000

n1=$GMPT_DATA_LOOKUP_VOLUME
n2=$GMPT_ETH_DATA_VOLUME
n3=$GMPT_WIKI_DATA_VOLUME
n4=$GMPT_YCSB_DATA_VOLUME

# modify the build
BUILD_PATH=./build

rm test_ycsb_lookup.log
rm test_wiki_lookup.log
rm test_eth_lookup.log
rm test_ycsb_insert.log
rm test_wiki_insert.log
rm test_eth_insert.log

for n in $(seq 8); do
    $BUILD_PATH/utils "--gtest_filter=Util.args" "--gtest_also_run_disabled_tests"
    echo test_ycsb_lookup...
    run_n_times 10 $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupYCSB" >> test_ycsb_lookup.log
    echo test_wiki_lookup...
    run_n_times 10 $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupWiki" >> test_wiki_lookup.log
    echo test_eth_lookup...
    run_n_times 10 $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupEthtxn" >> test_eth_lookup.log
    echo test_ycsb_insert...
    run_n_times 10 $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertYCSB" >> test_ycsb_insert.log
    echo test_wiki_insert...
    run_n_times 10 $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertWiki" >> test_wiki_insert.log
    echo test_eth_insert...
    run_n_times 10 $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertEthtxn" >> test_eth_insert.log

    n1=`expr $n1 / 2`
    n2=`expr $n2 / 2`
    n3=`expr $n3 / 2`
    n4=`expr $n4 / 2`
    export GMPT_DATA_LOOKUP_VOLUME=$n1
    export GMPT_ETH_DATA_VOLUME=$n2
    export GMPT_WIKI_DATA_VOLUME=$n3
    export GMPT_YCSB_DATA_VOLUME=$n4
done
