#!/bin/bash -e

run_n_times(){
    number=$1
    shift
    for n in $(seq $number); do
      $@
      sleep 2
    done
}

export GMPT_MODEL_DATA_VOLUME=400

n1=$GMPT_MODEL_DATA_VOLUME

# modify the build path
BUILD_PATH=./build
REPEAT=5

rm ./data/cluster.csv
rm ./data/uniform.csv
rm test_model.log

for n in $(seq 10); do
    $BUILD_PATH/utils "--gtest_filter=Util.args" "--gtest_also_run_disabled_tests"

    echo test_model...
    run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.Cluster" >> test_model.log
    run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.Uniform" >> test_model.log

    n1=`expr $n1 - 10 `
    echo $n1
    export GMPT_MODEL_DATA_VOLUME=$n1
done
