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

# modify the build path
BUILD_PATH=./build
GO_ETH_PATH=./go-ethereum/trie
REPEAT=10

rm test_ycsb_lookup.log
rm test_wiki_lookup.log
rm test_eth_lookup.log
rm test_ycsb_insert.log
rm test_wiki_insert.log
rm test_eth_insert.log

for n in $(seq 8); do
    $BUILD_PATH/utils "--gtest_filter=Util.args" "--gtest_also_run_disabled_tests"

    # echo test_ycsb_lookup...
    # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupYCSB" >> test_ycsb_lookup.log
    # pushd $GO_ETH_PATH
    # go test -run ^TestLookupYCSB$ -count=$REPEAT >> ../../test_ycsb_lookup.log
    # go test -run ^TestLookupYCSBParallel$ -count=$REPEAT >> ../../test_ycsb_lookup.log
    # popd

    # echo test_wiki_lookup...
    # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupWiki" >> test_wiki_lookup.log
    # pushd $GO_ETH_PATH
    # go test -run ^TestLookupWiki$ -count=$REPEAT >> ../../test_wiki_lookup.log
    # go test -run ^TestLookupWikiParallel$ -count=$REPEAT >> ../../test_wiki_lookup.log
    # popd

    # echo test_eth_lookup...
    # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupEthtxn" >> test_eth_lookup.log
    # pushd $GO_ETH_PATH
    # go test -run ^TestLookupEthtxn$ -count=$REPEAT >> ../../test_eth_lookup.log
    # go test -run ^TestLookupEthtxnParallel$ -count=$REPEAT >> ../../test_eth_lookup.log
    # popd

    echo test_ycsb_insert...
    run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertYCSB" >> test_ycsb_insert.log
    pushd $GO_ETH_PATH
    go test -run ^TestInsertYCSB$ -count=$REPEAT >> ../../test_ycsb_insert.log
    popd
    
    echo test_wiki_insert...
    run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertWiki" >> test_wiki_insert.log
    pushd $GO_ETH_PATH
    go test -run ^TestInsertWiki$ -count=$REPEAT >> ../../test_wiki_insert.log
    popd

    echo test_eth_insert...
    run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertEthtxn" >> test_eth_insert.log
    pushd $GO_ETH_PATH
    go test -run ^TestInsertEthtxn$ -count=$REPEAT >> ../../test_eth_insert.log
    popd
    
    n1=`expr $n1 / 2`
    n2=`expr $n2 / 2`
    n3=`expr $n3 / 2`
    n4=`expr $n4 / 2`
    export GMPT_DATA_LOOKUP_VOLUME=$n1
    export GMPT_ETH_DATA_VOLUME=$n2
    export GMPT_WIKI_DATA_VOLUME=$n3
    export GMPT_YCSB_DATA_VOLUME=$n4
done
