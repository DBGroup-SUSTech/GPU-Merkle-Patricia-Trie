#!/bin/bash -e

run_n_times(){
    number=$1
    shift
    for n in $(seq $number); do
      $@
      sleep 2
    done
}

export GMPT_KEYTYPE_LEN=6
export GMPT_KEYTYPE_STEP=450000
REPEAT=10

n1=$GMPT_KEYTYPE_LEN
n2=$GMPT_KEYTYPE_STEP

BUILD_PATH=./build
rm test_model_fitting.log
rm model_data.csv

for m in $(seq 80); do
  for n in $(seq 20); do
    echo test_key_type...
    run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.ModelFitting" >> test_model_fitting.log
    n1=`expr $n1 + 2`
    echo $n1
    export GMPT_KEYTYPE_LEN=$n1
  done
  n2=`expr $n2 + 1000`
  echo $n2
  n1=6
  export GMPT_KEYTYPE_LEN=$n1
  export GMPT_KEYTYPE_STEP=$n2
done

