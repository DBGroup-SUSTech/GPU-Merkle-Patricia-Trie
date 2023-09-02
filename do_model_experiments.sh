#!/bin/bash -e

run_n_times(){
    number=$1
    shift
    for n in $(seq $number); do
      $@
      sleep 2
    done
}

export GMPT_MODEL_DATA_VOLUME=1048576
# export GMPT_MODEL_CLUSTER_NUM=5
export GMPT_MODEL_DATA_SIZE=640000

n1=$GMPT_MODEL_DATA_VOLUME
# n2=$GMPT_MODEL_CLUSTER_NUM
n3=$GMPT_MODEL_DATA_SIZE

# modify the build path
BUILD_PATH=./build
REPEAT=5

# rm ./data/cluster*
# rm ./data/multi_cluster*
# rm ./data/uniform*
# rm ./data/rangetriesize*
# # rm ./data/uniformnew*
# # rm ./data/rangetriesizecluster*
# rm ./data/smalluniform*
# rm ./data/mediumuniform*
# rm ./data/largeuniform*
# rm ./data/uniform*

rm test_model.log

# for (( a = 1; a <= 10; a++ )) do
#   echo test_uniform...
#   run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.SmallUniform" >> test_model.log
#   n1=`expr $n1 + 1 `
#   echo $n1
#   export GMPT_MODEL_DATA_VOLUME=$n1
# done

# export GMPT_MODEL_DATA_VOLUME=5
# n1=$GMPT_MODEL_DATA_VOLUME
# export GMPT_MODEL_DATA_SIZE=80000

for (( a = 1; a <= 20; a++)) do 
  echo test_cluster...
  run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.Cluster" >> test_model.log
  n1=`expr $n1 / 2 `
  echo $n1
  export GMPT_MODEL_DATA_VOLUME=$n1
done

# for (( a = 1; a <= 20; a++ )) do
#   echo test_uniform...
#   run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.SmallUniform" >> test_model.log
#   n1=`expr $n1 + 1 `
#   echo $n1
#   export GMPT_MODEL_DATA_VOLUME=$n1
# done

# export GMPT_MODEL_DATA_VOLUME=5
# n1=$GMPT_MODEL_DATA_VOLUME
# export GMPT_MODEL_DATA_SIZE=10000

# for (( a = 1; a <= 30; a++ )) do
#   echo test_uniform...
#   run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.SmallUniform" >> test_model.log
#   n1=`expr $n1 + 1 `
#   echo $n1
#   export GMPT_MODEL_DATA_VOLUME=$n1
# done
# for (( a = 1; a <=20; a++ )) do
#   echo test_medium_and_large_uniform...
#   run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.MediumUniform" >> test_model.log
#   run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.LargeUniform" >> test_model.log
#   n1=`expr $n1 / 2 `
#   echo $n1
#   export GMPT_MODEL_DATA_VOLUME=$n1
# done

# for (( b = 1; b <= 40; b++ )) do
#   echo test_small_uniform...
#   run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.SmallUniform" >> test_model.log
#   n3=`expr $n3 - 1 `
#   echo $n3
#   export GMPT_MODEL_DATA_SIZE=$n3
# done

# for (( c = 1; c <= 8; c++ )) do
#   for (( a = 1; a <= 20; a++ )) do
#       $BUILD_PATH/utils "--gtest_filter=Util.args" "--gtest_also_run_disabled_tests"

#       echo test_model...
#       run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.RangeTrieSize" >> test_model.log
#       # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.Uniform" >> test_model.log
#       # run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.RangeTrieCluster" >> test_model.log

#       # echo test_multi_cluster...
#       # for (( b = 1; b <= 5; b++ )) do
#       #   run_n_times $REPEAT $BUILD_PATH/experiments "--gtest_filter=EXPERIMENTS.MultiCluster" >> test_model.log
#       #   n2=`expr $n2 - 1`
#       #   echo $n2
#       #   export GMPT_MODEL_CLUSTER_NUM=$n2
#       # done

#       # export GMPT_MODEL_CLUSTER_NUM=5
#       # n2=$GMPT_MODEL_CLUSTER_NUM
#       n1=`expr $n1 / 2 `
#       echo $n1
#       export GMPT_MODEL_DATA_VOLUME=$n1
#   done
#   export GMPT_MODEL_DATA_VOLUME=1048576
#   n1=$GMPT_MODEL_DATA_VOLUME
#   n3=`expr $n3 / 2 `
#   echo $n3
#   export GMPT_MODEL_DATA_SIZE=$n3
# done