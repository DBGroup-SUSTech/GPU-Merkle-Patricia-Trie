#!/bin/bash -e

run_n_times(){
    number=$1
    shift
    for n in $(seq $number); do
      $@
      sleep 2
    done
}


# modify the build path
GO_ETH_PATH=./go-ethereum/miner
GO_REPEAT=10
OUTPUT_FILE=case_study-geth-to-640000.log

rm $OUTPUT_FILE
# TODO: geth's code can only modified 
l="5000 40000 160000 320000 640000"
for n in $l; do
    echo test...$n...
    pushd $GO_ETH_PATH
    for i in $(seq 10); do
      CASE_ETHT_DATA_VOLUME=$n go test -run ^TestGMPTTransactionProcessing$ -count=1 -timeout 0 >> ../../$OUTPUT_FILE
      sleep 2
    done
    popd
done
