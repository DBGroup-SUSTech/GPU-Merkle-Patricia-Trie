# GPU Merkle Patricia Trie

Note: We are working on completing this documents.

## Dependencies
* crypto++ 8.7
* libxml2
* gtest
* golang

## Submodules
We have two externel git submodules. One includes our modified go-ethereum implementation and experiments. Another includes our modified ycsb generator. They are forked from open-sourced branch. 

For anonymity, we removed the `.gitmodules` and also make them anonymous. Please download [Geth](https://anonymous.4open.science/r/Accelerating-Merkle-Patricia-Trie-with-GPU-Geth/) and [YCSB](https://anonymous.4open.science/r/Accelerating-Merkle-Patricia-Trie-with-GPU-YCSB/) into current folder before you start to build.
## Data
You can load the ycsb and wiki dataset by runing our script
```sh
bash download_data.sh
```
Ethereum transaction dataset is a financial dataset. It should be downloaded from [Google BigQuery](https://cloud.google.com/blog/products/data-analytics/ethereum-bigquery-public-dataset-smart-contract-analytics) by following query.
```sql
select *
from bigquery-public-data.crypto_ethereum.transactions 
where 
  EXTRACT(date FROM block_timestamp) between "2022-11-15" and "2022-12-29"
```
The results need to be exported to one or multiple csv files and saved in `/ethereum/transactions/` folder.
## Build
```sh
mkdir build
cd build
cmake ../mpt-with-compress/
cmake --build . -j
```

For Ethereum transaction processing case study:
```sh
bash geth_install_libgmpt.sh
```

## Run
### Experiments
You can run unit tests in [go-ethereum/trie/experiments_test.go](./go-ethereum/trie/experiments_test.go) and 
``` sh
bash do_all_experiments.sh
```
* Run go-ethereum experients in [go-ethereum/trie/experiments_test.go](./go-ethereum/trie/experiments_test.go)
### Profiling
You can run the experiments and generate profiling report by [profile.sh](./profile.sh). The Nsight Compute report will be stored in `./profile/`. If you'd like a csv version, just run [convert_to_csv.sh](./convert_to_csv.sh) and you will get all csv reports in `./profile/`.

### Case study: Applied to Go-ethereum
We integrate GPU MPT into go-ethereum using [cgo](https://golang.google.cn/pkg/runtime/cgo/).
#### Install
First make sure the `gmpt` target is compiled. 
Then run [geth_install_libgmpt.sh](./geth_install_libgmpt.sh) to install the library and header files into go-ethereum's folder.
#### Configure geth to use GPU MPT
- [ ] TODO

## Code Structure
### Overview
```
.
├── dataset           # created by [download_data.sh](./download_data.sh)
├── go-ethereum       # forked from official go-ethereum. downloaded by git submodule
├── YCSB-C            # YCSB workload generator. downloaded by git submodule
├── mpt-with-compress # GPU MPT implementations
└── mpt-no-compress   # You can ignore it
```
### Key Algorithms
* **PhaseNU**: `GpuMPT::Compress::MPT::puts_2phase_with_valuehp` in [./mpt-with-compress/include/mpt/gpu_mpt.cuh](./mpt-with-compress/include/mpt/gpu_mpt.cuh)
* **LockNU**: `GpuMPT::Compress::MPT::puts_latching_with_valuehp_v2` in [./mpt-with-compress/include/mpt/gpu_mpt.cuh](./mpt-with-compress/include/mpt/gpu_mpt.cuh)
* **PhaseHC**: `GpuMPT::Compress::MPT::hash_onepass_v2` in [./mpt-with-compress/include/mpt/gpu_mpt.cuh](./mpt-with-compress/include/mpt/gpu_mpt.cuh)
### Extension of LockNU
* **LockNU on B-Tree**: `GpuBTree::OLC::BTree::puts_olc_with_vsize` in [./mpt-with-compress/include/skiplist/gpu_skiplist.cuh](./mpt-with-compress/include/skiplist/gpu_skiplist.cuh)
* **CPU baseline B-Tree**: `CpuBTree::BTree::puts_baseline` in [./mpt-with-compress/include/btree/cpu_btree.cuh](./mpt-with-compress/include/btree/cpu_btree.cuh)
* **LockNU on SkipList**: `GpuSkiplist::SkipList::puts_olc_with_ksize` in [./mpt-with-compress/include/skiplist/gpu_skiplist.cuh](./mpt-with-compress/include/skiplist/gpu_skiplist.cuh)
* **CPU baseline SkipList**: `CpuSkiplist::SkipList::puts_baseline` in [./mpt-with-compress/include/skiplist/cpu_skiplist.cuh] (./mpt-with-compress/include/skiplist/cpu_skiplist.cuh)
### Key experiments
Please refer to [./do_all_experiments.sh](./do_all_experiments.sh), [./go-ethereum/trie/experiments_test.go](./go-ethereum/trie/experiments_test.go) and [./go-ethereum/miner/benchmark_test.go](./go-ethereum/miner/benchmark_test.go).
