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
export GMPT_YCSB_DATA_VOLUME=100000
export GMPT_DATA_LOOKUP_VOLUME=1280000

export GMPT_THREAD_NUM=4


n1=$GMPT_DATA_LOOKUP_VOLUME
n2=$GMPT_ETH_DATA_VOLUME
n3=$GMPT_WIKI_DATA_VOLUME
n4=$GMPT_YCSB_DATA_VOLUME
n5=$GMPT_THREAD_NUM

# modify the build path
BUILD_PATH=./build
GO_ETH_PATH=./go-ethereum/trie
REPEAT=10
GO_REPEAT=10

rm test_ycsb_lookup.log
rm test_wiki_lookup.log
rm test_eth_lookup.log
rm test_ycsb_insert.log
rm test_wiki_insert.log
rm test_eth_insert.log

rm ./data/*

# for (( b = 1; b <=6; b++ )) do
#     # echo test_ycsb_insert...
#     # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertYCSB" >> test_ycsb_insert.log
#     # pushd $GO_ETH_PATH
#     # go test -run ^TestInsertYCSB$ -count=$GO_REPEAT >> ../../test_ycsb_insert.log
#     # popd
rm test_eth_warp.log

# for n in $(seq 8); do
    # $BUILD_PATH/utils "--gtest_filter=Util.args" "--gtest_also_run_disabled_tests"

#     echo test_wiki_insert...
#     run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertWiki" >> test_wiki_insert.log
#     pushd $GO_ETH_PATH
#     go test -run ^TestInsertWiki$ -count=$GO_REPEAT >> ../../test_wiki_insert.log
#     popd
#     n5=`expr $n5 \* 2`
#     export GMPT_THREAD_NUM=$n5
#   done


for (( a = 1; a <=8; a++ )) do
    # echo test_wiki_insert...
    # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertWiki" >> test_wiki_insert.log
    # pushd $GO_ETH_PATH
    # go test -run ^TestInsertWiki$ -count=$GO_REPEAT >> ../../test_wiki_insert.log
    # popd

    # echo test_eth_insert...
    # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertEthtxn" >> test_eth_insert.log
    # pushd $GO_ETH_PATH
    # go test -run ^TestInsertEthtxn$ -count=$GO_REPEAT >> ../../test_eth_insert.log
    # popd

  # echo test_wiki_insert...
  # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertWiki" >> test_wiki_insert.log
  # pushd $GO_ETH_PATH
  # go test -run ^TestInsertWiki$ -count=$GO_REPEAT >> ../../test_wiki_insert.log
  # popd
  echo test_ycsb_lookup...
  run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupYCSB" >> test_ycsb_lookup.log
  pushd $GO_ETH_PATH
  # go test -run ^TestLookupYCSB$ -count=$GO_REPEAT >> ../../test_ycsb_lookup.log
  go test -run ^TestLookupYCSBParallel$ -count=$GO_REPEAT >> ../../test_ycsb_lookup.log
  popd

  n1=`expr $n1 / 2`
  n2=`expr $n2 / 2`
  n3=`expr $n3 / 2`
  n4=`expr $n4 + 100000`
  export GMPT_DATA_LOOKUP_VOLUME=$n1
  export GMPT_ETH_DATA_VOLUME=$n2
  export GMPT_WIKI_DATA_VOLUME=$n3
  export GMPT_YCSB_DATA_VOLUME=$n4
done
for n in $(seq 6); do
    $BUILD_PATH/utils "--gtest_filter=Util.args" "--gtest_also_run_disabled_tests"

#     # echo test_ycsb_lookup...
#     # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupYCSB" >> test_ycsb_lookup.log
#     # pushd $GO_ETH_PATH
#     # go test -run ^TestLookupYCSB$ -count=$GO_REPEAT >> ../../test_ycsb_lookup.log
#     # go test -run ^TestLookupYCSBParallel$ -count=$GO_REPEAT >> ../../test_ycsb_lookup.log
#     # popd

#     # echo test_wiki_lookup...
#     # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupWiki" >> test_wiki_lookup.log
#     # pushd $GO_ETH_PATH
#     # go test -run ^TestLookupWiki$ -count=$GO_REPEAT>> ../../test_wiki_lookup.log
#     # go test -run ^TestLookupWikiParallel$ -count=$GO_REPEAT >> ../../test_wiki_lookup.log
#     # popd

#     # echo test_eth_lookup...
#     # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupEthtxn" >> test_eth_lookup.log
#     # pushd $GO_ETH_PATH
#     # go test -run ^TestLookupEthtxn$ -count=$GO_REPEAT>> ../../test_eth_lookup.log
#     # go test -run ^TestLookupEthtxnParallel$ -count=$GO_REPEAT >> ../../test_eth_lookup.log
#     # popd

#     echo test_ycsb_insert...
#     run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertYCSB" >> test_ycsb_insert.log
#     pushd $GO_ETH_PATH
#     go test -run ^TestInsertYCSB$ -count=$GO_REPEAT >> ../../test_ycsb_insert.log
#     popd
    
#     # echo test_wiki_insert...
#     # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertWiki" >> test_wiki_insert.log
#     # pushd $GO_ETH_PATH
#     # go test -run ^TestInsertWiki$ -count=$GO_REPEAT >> ../../test_wiki_insert.log
#     # popd

#     # echo test_eth_insert...
#     # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertEthtxn" >> test_eth_insert.log
#     # pushd $GO_ETH_PATH
#     # go test -run ^TestInsertEthtxn$ -count=$GO_REPEAT >> ../../test_eth_insert.log
#     # popd
    
    echo test_eth_warp_hash...
    run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.WarpETHT" >> test_eth_warp-100k-600k.log
    
    n1=`expr $n1 / 2`
    n2=`expr $n2 / 2`
    n3=`expr $n3 / 2`
    n4=`expr $n4 + 100000`

    export GMPT_DATA_LOOKUP_VOLUME=$n1
    export GMPT_ETH_DATA_VOLUME=$n2
    export GMPT_WIKI_DATA_VOLUME=$n3
    export GMPT_YCSB_DATA_VOLUME=$n4
done

# bash do_rw_experiments.sh