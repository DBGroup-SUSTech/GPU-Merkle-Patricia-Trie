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

for n in $(seq 8); do
    /home/ymx/ccpro/bench/GPU-Merkle-Patricia-Trie/mpt-with-compress/build/utils "--gtest_filter=Util.args" "--gtest_also_run_disabled_tests"
    run_n_times 10 /home/ymx/ccpro/bench/GPU-Merkle-Patricia-Trie/mpt-with-compress/build/trie "--gtest_filter=TrieV2.LookupYCSBBench" "--gtest_also_run_disabled_tests" >> test_ycsb_lookup.log
    run_n_times 10 /home/ymx/ccpro/bench/GPU-Merkle-Patricia-Trie/mpt-with-compress/build/trie "--gtest_filter=TrieV2.LookupWikiBench" "--gtest_also_run_disabled_tests" >> test_wiki_lookup.log
    run_n_times 10 /home/ymx/ccpro/bench/GPU-Merkle-Patricia-Trie/mpt-with-compress/build/trie "--gtest_filter=TrieV2.LookupEthtxnBench" "--gtest_also_run_disabled_tests" >> test_eth_lookup.log
    run_n_times 10 /home/ymx/ccpro/bench/GPU-Merkle-Patricia-Trie/mpt-with-compress/build/trie "--gtest_filter=TrieV2.ETEInsertYCSBBench" "--gtest_also_run_disabled_tests" >> test_ycsb_insert.log
    run_n_times 10 /home/ymx/ccpro/bench/GPU-Merkle-Patricia-Trie/mpt-with-compress/build/trie "--gtest_filter=TrieV2.ETEInsertWikiBench" "--gtest_also_run_disabled_tests" >> test_wiki_insert.log
    run_n_times 10 /home/ymx/ccpro/bench/GPU-Merkle-Patricia-Trie/mpt-with-compress/build/trie "--gtest_filter=TrieV2.ETEInsertEthtxnBench" "--gtest_also_run_disabled_tests" >> test_eth_insert.log

    n1=`expr $n1 / 2`
    n2=`expr $n2 / 2`
    n3=`expr $n3 / 2`
    n4=`expr $n4 / 2`
    export GMPT_DATA_LOOKUP_VOLUME=$n1
    export GMPT_ETH_DATA_VOLUME=$n2
    export GMPT_WIKI_DATA_VOLUME=$n3
    export GMPT_YCSB_DATA_VOLUME=$n4
done
