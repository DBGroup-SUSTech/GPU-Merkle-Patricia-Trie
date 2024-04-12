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


OUTPUT_FILE=ycsb.log
# modify the build path
BUILD_PATH=build
GO_ETH_PATH=./go-ethereum/trie
REPEAT=5
GO_REPEAT=5

rm $OUTPUT_FILE

l="800000"
for n in $l; do
    echo test...$n...
    export GMPT_YCSB_DATA_VOLUME=$n
    $BUILD_PATH/utils "--gtest_filter=Util.args" "--gtest_also_run_disabled_tests"
    run_n_times $REPEAT numactl --membind 0 $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.InsertYCSB" >> $OUTPUT_FILE
    pushd $GO_ETH_PATH
      go test -run ^TestInsertYCSB$ -count=$GO_REPEAT -timeout 0 >> ../../$OUTPUT_FILE
    # for i in $(seq 10); do
    #   CASE_ETHT_DATA_VOLUME=$n go test -run ^TestGMPTTransactionProcessing$ -count=1 -timeout 0 >> ../../$OUTPUT_FILE
    #   sleep 2
    # done
    popd
done