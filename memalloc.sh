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
BUILD_PATH=build
# GO_ETH_PATH=./go-ethereum/trie
REPEAT=5
# GO_REPEAT=30

# ncu
NCU=/usr/local/cuda/bin/ncu

rm -rf profile
rm test_alloc.log
# rm ./data/*

mkdir -p profile

# for n in $(seq 8); do
#    run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.PutPhase" >> test_alloc.log
#    n1=`expr $n1 / 2`
#    n2=`expr $n2 / 2`
#    n3=`expr $n3 / 2`
#    n4=`expr $n4 / 2`
#    export GMPT_DATA_LOOKUP_VOLUME=$n1
#    export GMPT_ETH_DATA_VOLUME=$n2
#    export GMPT_WIKI_DATA_VOLUME=$n3
#    export GMPT_YCSB_DATA_VOLUME=$n4
# done

# export GMPT_WIKI_DATA_VOLUME=320000
# export GMPT_ETH_DATA_VOLUME=640000
# export GMPT_YCSB_DATA_VOLUME=1280000
# export GMPT_DATA_LOOKUP_VOLUME=1280000

# n1=$GMPT_DATA_LOOKUP_VOLUME
# n2=$GMPT_ETH_DATA_VOLUME
# n3=$GMPT_WIKI_DATA_VOLUME
# n4=$GMPT_YCSB_DATA_VOLUME

for n in $(seq 8); do
  # sudo GMPT_ETH_DATA_VOLUME=$GMPT_ETH_DATA_VOLUME       $NCU -s 5 -f -o profile/InsertEth$GMPT_ETH_DATA_VOLUME $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertEthtxn"
  # sudo GMPT_DATA_LOOKUP_VOLUME=$GMPT_DATA_LOOKUP_VOLUME $NCU -s 5 -f -o profile/LookupEth$GMPT_DATA_LOOKUP_VOLUME $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupEthtxn" 
  
  # sudo GMPT_YCSB_DATA_VOLUME=$GMPT_YCSB_DATA_VOLUME     $NCU -s 5 -f -o profile/InsertYCSB$GMPT_YCSB_DATA_VOLUME $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertYCSB"
  # sudo GMPT_DATA_LOOKUP_VOLUME=$GMPT_DATA_LOOKUP_VOLUME $NCU -s 5 -f -o profile/LookupYCSB$GMPT_DATA_LOOKUP_VOLUME $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupYCSB" 

  # sudo GMPT_WIKI_DATA_VOLUME=$GMPT_WIKI_DATA_VOLUME     $NCU -s 5 -f -o profile/InsertWiki$GMPT_WIKI_DATA_VOLUME $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertWiki"
  # sudo GMPT_DATA_LOOKUP_VOLUME=$GMPT_DATA_LOOKUP_VOLUME $NCU -s 5 -f -o profile/LookupWiki$GMPT_DATA_LOOKUP_VOLUME $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LookupWiki" 

  sudo GMPT_ETH_DATA_VOLUME=$GMPT_ETH_DATA_VOLUME  LD_LIBRARY_PATH=$LD_LIBRARY_PATH    $NCU  --section MemoryWorkloadAnalysis -f -o profile/memEth$GMPT_ETH_DATA_VOLUME $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.PutPhase" >> test_alloc.log

  n1=`expr $n1 / 2`
  n2=`expr $n2 / 2`
  n3=`expr $n3 / 2`
  n4=`expr $n4 / 2`
  export GMPT_DATA_LOOKUP_VOLUME=$n1
  export GMPT_ETH_DATA_VOLUME=$n2
  export GMPT_WIKI_DATA_VOLUME=$n3
  export GMPT_YCSB_DATA_VOLUME=$n4
done