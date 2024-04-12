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
export GMPT_KEYTYPE_LEN=40  # hex len = $ * 2 + 1
export GMPT_KEYTYPE_NUM=320000
n1=$GMPT_TRIESIZE
n2=$GMPT_KEYTYPE_LEN

BUILD_PATH=./build
GO_ETH_PATH=./go-ethereum/trie
REPEAT=10

rm test_sup_experiments.log
rm test_dense.log
rm test_sparse.log

for n in $(seq 5); do
  echo test_key_type...
  run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.KeyTypeSparse" >> test_sparse.log
  run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.KeyTypeDense" >> test_dense.log
  n2=`expr $n2 / 2`
  echo $n2
  export GMPT_KEYTYPE_LEN=$n2
done

# for n in $(seq 7); do
#     # $BUILD_PATH/utils "--gtest_filter=Util.args" "--gtest_also_run_disabled_tests"

#     echo test_trie_size...
#     run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.TrieSizeEthtxn" >> test_sup_experiments.log
#     pushd $GO_ETH_PATH
#     go test -run ^TestEthTrieSize$ -count=$REPEAT >> ../../test_sup_experiments.log
#     popd
    
#     n1=`expr $n1 + 40000 `
#     echo $n1
#     export GMPT_TRIESIZE=$n1
# done

# echo test_async...
# run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.AsyncMemcpyEthtxn" >> test_sup_experiments.log
