# GPU Merkle Patricia Trie
## Dependencies
* crypto++ 8.7
* libxml2
* gtest
* golang

## Submodules
We have two externel git submodules. One includes our modified go-ethereum implementation and experiments. Another includes our modified ycsb generator. They are forked from open-sourced branch. 

For anonymity, we removed the `.gitmodules` and also make them anonymous. Please download [Geth](https://anonymous.4open.science/r/Accelerating-Merkle-Patricia-Trie-with-GPU-Geth/) and [YCSB](https://anonymous.4open.science/r/Accelerating-Merkle-Patricia-Trie-with-GPU-YCSB/) into current folder before you start to build.
### Data
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
### Build
```sh
mkdir build
cd build
cmake ../mpt-with-compress/
cmake --build . -j
```
### Run
You can run unit tests in [go-ethereum/trie/experiments_test.go](./go-ethereum/trie/experiments_test.go) and 
``` sh
bash do_all_experiments.sh
```
* Run go-ethereum experients in [go-ethereum/trie/experiments_test.go](./go-ethereum/trie/experiments_test.go)