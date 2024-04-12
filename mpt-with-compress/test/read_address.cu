#include "bench/ethtxn.cuh"
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <tbb/tbb.h>
#include "mpt/cpu_mpt.cuh"
#include "mpt/gpu_mpt.cuh"
#include "util/experiments.cuh"


const uint8_t **get_values_hps(int n, const int64_t *values_bytes_indexs,
                               const uint8_t *values_bytes)
{
    const uint8_t **values_hps = new const uint8_t *[n];
    for (int i = 0; i < n; ++i)
    {
        values_hps[i] = util::element_start(values_bytes_indexs, i, values_bytes);
    }
    return values_hps;
}

void keys_bytes_to_hexs(const uint8_t *keys_bytes, int *keys_bytes_indexs,
                        int n, const uint8_t *&keys_hexs,
                        int *&keys_hexs_indexs)
{
    int keys_bytes_size = util::elements_size_sum(keys_bytes_indexs, n);
    int keys_hexs_size = keys_bytes_size * 2 + n;

    uint8_t *hexs = new uint8_t[keys_hexs_size]{};
    int *hexs_indexs = new int[2 * n]{};

    for (int next_key_hexs = 0, i = 0; i < n; ++i)
    {
        const uint8_t *key_bytes =
            util::element_start(keys_bytes_indexs, i, keys_bytes);
        int key_bytes_size = util::element_size(keys_bytes_indexs, i);

        int key_hexs_size =
            util::key_bytes_to_hex(key_bytes, key_bytes_size, hexs + next_key_hexs);

        hexs_indexs[2 * i] = next_key_hexs;
        hexs_indexs[2 * i + 1] = next_key_hexs + key_hexs_size - 1;

        next_key_hexs += key_hexs_size; // write to next elements
    }

    keys_hexs = hexs;
    keys_hexs_indexs = hexs_indexs;
}

void calc_size(std::string file_name, int &key_size, int &value_size, int &n)
{
    std::ifstream file;
    using namespace bench::ethtxn;
    std::string sample = "108532917307676136273000";
    int leaf_value_size = sample.size();
    file.open(file_name, std::ios::in);
    if (!file)
    {
        printf("no file\n");
        assert(false);
    }
    std::string line;
    while (std::getline(file, line, '\n'))
    {
        if (line == "leafs")
        {
            break;
        }
        key_size += line.size();
        value_size += 32;
        n++;
    }
    while (std::getline(file, line, '\n'))
    {
        key_size += line.size() + 1;
        value_size += sample.size();
        n++;
    }
    file.close();
}

void fetch_and_generate_ethaddress_data(std::string file_name, const uint8_t *&keys_hexs, int *&keys_hexs_indexs, uint8_t *&values_bytes, int64_t *&values_bytes_indexs, int &n)
{
    std::ifstream file;
    using namespace bench::ethtxn;
    std::string sample = "108532917307676136273000";
    int leaf_value_size = sample.size();
    int keys_hexs_size = 0;
    int values_bytes_size = 0;

    calc_size(file_name, keys_hexs_size, values_bytes_size, n);
    file.open(file_name, std::ios::in);
    if (!file)
    {
        printf("no file\n");
        assert(false);
    }

    uint8_t *hexs = new uint8_t[keys_hexs_size]{};
    uint8_t *values = new uint8_t[values_bytes_size]{};
    int *hexs_indexs = new int[2 * n]{};
    int64_t *values_indexs = new int64_t[2 * n]{};
    std::string line;
    int key_length = 0;
    int value_length = 0;
    int i = 0;
    while (std::getline(file, line, '\n'))
    {
        // const char *split = ":";
        std::string key;
        std::stringstream ss(line);
        std::getline(ss, key, ',');
        if (key == "leafs")
        {
            break;
        }

        uint8_t *hex_array = new uint8_t[key.size()];
        for (int b = 0; b < key.size(); b++)
        {
            // Convert the character to a uint8_t value
            uint8_t byteValue = static_cast<uint8_t>(std::stoi(std::string(1, key[b]), nullptr, 16));
            hex_array[b] = byteValue;
        }

        memcpy(hexs + key_length, hex_array, key.size());
        hexs_indexs[2 * i] = key_length;
        key_length += key.size();
        hexs_indexs[2 * i + 1] = key_length - 1;

        free(hex_array);

        // int value_size = util::align_to<8>(static_cast<int>(value.size()));
        memset(values + value_length, 0, 32);
        values_indexs[2 * i] = value_length;
        value_length += 32;
        values_indexs[2 * i + 1] = value_length - 1;
        i++;
    }
    while (std::getline(file, line, '\n'))
    {
        std::string key;
        std::stringstream ss(line);
        std::getline(ss, key, ',');
        uint8_t *hex_array = new uint8_t[key.size()];
        for (int b = 0; b < key.size(); b++)
        {
            // Convert the character to a uint8_t value
            uint8_t byteValue = static_cast<uint8_t>(std::stoi(std::string(1, key[b]), nullptr, 16));
            hex_array[b] = byteValue;
        }

        memcpy(hexs + key_length, hex_array, key.size());
        hexs_indexs[2 * i] = key_length;
        key_length += key.size();
        hexs_indexs[2 * i + 1] = key_length - 1;

        free(hex_array);

        memset(values + value_length, 0, leaf_value_size);
        values_indexs[2 * i] = value_length;
        value_length += leaf_value_size;
        values_indexs[2 * i + 1] = value_length - 1;
        i++;
    }
    // Close the file

    keys_hexs = hexs;
    keys_hexs_indexs = hexs_indexs;
    values_bytes = values;
    values_bytes_indexs = values_indexs;

    file.close();
}

using kvs_type = std::vector<std::pair<std::string, std::string>>;
void random_transaction_generator(int n, kvs_type &addresses, std::vector<int> &txns_addresses_indexs)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, addresses.size() - 1);

    for (int i = 0; i < n; i++)
    {
        int src_idx = dis(gen);
        int dst_idx = dis(gen);
        txns_addresses_indexs.push_back(src_idx);
        txns_addresses_indexs.push_back(dst_idx);
    }
    // sort and make unique
    std::sort(txns_addresses_indexs.begin(), txns_addresses_indexs.end());
    txns_addresses_indexs.erase(std::unique(txns_addresses_indexs.begin(), txns_addresses_indexs.end()), txns_addresses_indexs.end());
}

void get_shared_prefix(std::string str1, std::string str2, int &prefix)
{
    int i = 0;
    while (i < str1.size() && i < str2.size() && str1[i] == str2[i])
    {
        i++;
    }
    prefix = i;
}

int key_transform(std::string o_key, uint8_t *out)
{
    std::string key;
    key = bench::ethtxn::hex_to_string(o_key);

    memcpy(out, (uint8_t *)key.c_str(), key.size());
    return key.size();
}

void generate_txn_reciept(const uint8_t *&keys_hexs, int *& keys_indexs, uint8_t *&values_bytes, int64_t *&values_indexs, int n) {
    int key_byte_size = sizeof(int);
    int value_size =4;
    uint8_t * keys_bytes = new uint8_t[n * key_byte_size];
    int * keys_bytes_indexs = new int[n * 2];
    values_bytes = new uint8_t[n * 4];
    memset(values_bytes, 0, n * 4);
    values_indexs = new int64_t[n * 2];

    for (int i=0;i< n;i++) {
        int key = i;
        memcpy(keys_bytes + i * key_byte_size, &key, key_byte_size);
        keys_bytes_indexs[2 * i] = i * key_byte_size;
        keys_bytes_indexs[2 * i + 1] = i * key_byte_size + key_byte_size - 1;

        values_indexs[2 * i] = i * value_size;
        values_indexs[2 * i + 1] = i * value_size + value_size - 1;
    }

    keys_bytes_to_hexs(keys_bytes, keys_bytes_indexs, n, keys_hexs, keys_indexs);

    delete[] keys_bytes;
    delete[] keys_bytes_indexs;
}

TEST(ethtxn, evaluatetxntriesize){
    int insert_num = 320000;
    // std::string file_name = "/home/ymx/ccnpro/GPU-Merkle-Patricia-Trie/ethaddress" + std::to_string(txn_num) + ".csv";
    const uint8_t *keys_hexs;
    int *keys_hexs_indexs;
    uint8_t *values_bytes;
    int64_t *values_bytes_indexs;

    generate_txn_reciept(keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs, insert_num);
    auto values_hps = get_values_hps(insert_num, values_bytes_indexs, values_bytes);
    int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
    int keys_indexs_size = util::indexs_size_sum(insert_num);
    int64_t values_bytes_size =
        util::elements_size_sum(values_bytes_indexs, insert_num);
    int values_indexs_size = util::indexs_size_sum(insert_num);
    int values_hps_size = insert_num;
    std::vector<std::string> columns = {"method", "data_num", "throughput"};

    exp_util::CSVDataRecorder insert_recorder(columns, "./data/insert_ycsb_thread.csv");

    {
        CHECK_ERROR(cudaDeviceReset());
        CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
        CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
        CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
        CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
        CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
        GPUHashMultiThread::load_constants();
        GpuMPT::Compress::MPT gpu_mpt_olc;
        auto [d_hash_nodes, hash_nodes_num] =
            gpu_mpt_olc.puts_latching_with_valuehp_v2_with_record(
                keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
                values_hps, insert_num, insert_recorder, insert_num);
        gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    }

    {
        CHECK_ERROR(cudaDeviceReset());
        CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
        CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
        CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
        CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
        CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
        GPUHashMultiThread::load_constants();
        GpuMPT::Compress::MPT gpu_mpt_two;
        auto [d_hash_nodes, hash_nodes_num] =
            gpu_mpt_two.puts_2phase_with_valuehp_with_recorder(
                keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
                values_hps, insert_num, insert_recorder, insert_num);
        gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    }
}

TEST(ethtxn, address_memory_analysis)
{
    
    std::vector<std::pair<std::string, std::string>> kvs;
    std::vector<int> txns_addresses_indexs;
    bench::ethtxn::read_ethaddress_data("/ethereum/ethaddress.csv", kvs);
    std::cout << "kvs: " << kvs.size() << std::endl;

    // std::string test_key = kvs[0].first;
    // uint8_t * key_bytes = new uint8_t[100];
    // uint8_t * key_hexs = new uint8_t[100];

    // std::cout << "test key: " << test_key << std::endl;

    // int byte_size = key_transform(test_key, key_bytes);
    // int hex_size = util::key_bytes_to_hex(key_bytes, byte_size, key_hexs);

    // cutil::print_hex(key_hexs, hex_size);
    // int txn_num = arg_util::get_record_num(arg_util::Dataset::TXN_NUM);
    int txn_num = 1500;
    random_transaction_generator(txn_num, kvs, txns_addresses_indexs);
    std::cout << "txn addresses: " << txns_addresses_indexs.size() << std::endl;

    tbb::concurrent_vector<std::string> cached_addresses;

    for (int i = 0; i < txns_addresses_indexs.size(); i++)
    {
        int left_i, current_i, right_i;
        if (i == 0)
        {
            left_i = 0;
        }
        else
        {
            left_i = txns_addresses_indexs[i - 1] + 1;
        }
        current_i = txns_addresses_indexs[i];
        if (i == txns_addresses_indexs.size() - 1)
        {
            right_i = kvs.size();
        }
        else
        {
            right_i = txns_addresses_indexs[i + 1];
        }
        // std::cout << "left_i: " << left_i << " current_i: " << current_i << " right_i: " << right_i << std::endl;
        tbb::parallel_for(tbb::blocked_range<int>(left_i, right_i),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int j = r.begin(); j < r.end(); j++)
                              {
                                  if (j == current_i)
                                  {
                                      continue;
                                  }
                                  int prefix;
                                  get_shared_prefix(kvs[j].first, kvs[current_i].first, prefix);
                                  std::string branch = kvs[j].first.substr(0, prefix + 1);
                                  cached_addresses.push_back(branch);
                              }
                          });
        // std::cout << "cached addresses: " << cached_addresses.size() << std::endl;
    }
    tbb::parallel_sort(cached_addresses.begin(), cached_addresses.end());
    auto last = std::unique(cached_addresses.begin(), cached_addresses.end());
    cached_addresses.resize(std::distance(cached_addresses.begin(), last));
    std::cout << "cached addresses: " << cached_addresses.size() << std::endl;
    for (int i = 0; i < cached_addresses.size() - 1; i++)
    {
        int prefix;
        get_shared_prefix(cached_addresses[i], cached_addresses[i + 1], prefix);
        if (prefix >= cached_addresses[i].size())
        {
            // rm cached_addresses[i]
            cached_addresses[i] = "?";
        }
    }

    last = std::remove(cached_addresses.begin(), cached_addresses.end(), "?");
    cached_addresses.resize(std::distance(cached_addresses.begin(), last));
    std::cout << "cached addresses: " << cached_addresses.size() << std::endl;

    std::string out_filename = "ethaddress" + std::to_string(txn_num) + ".csv";
    std::ofstream out(out_filename);
    int sum = 0;
    for (auto &address : cached_addresses)
    {
        sum += address.size();
        out << address << std::endl;
    }
    std::cout << "avg length: " << sum / cached_addresses.size() << std::endl;
    out << "leafs" << std::endl;
    for (auto s : txns_addresses_indexs)
    {
        out << kvs[s].first << std::endl;
    }
    out.close();
}

TEST(ethtxn, memory_analysis)
{
    // int txn_num = arg_util::get_record_num(arg_util::Dataset::TXN_NUM);
    // std::string file_name = "/home/ymx/ccnpro/GPU-Merkle-Patricia-Trie/ethaddress" + std::to_string(txn_num) + ".csv";
    std::string file_name = "/home/ymx/ccnpro/GPU-Merkle-Patricia-Trie/ethaddress1500.csv";
    int txn_num = 1500;

    const uint8_t *keys_hexs;
    int *keys_hexs_indexs;
    uint8_t *values_bytes;
    int64_t *values_bytes_indexs;
    int insert_num = 0;
    fetch_and_generate_ethaddress_data(file_name, keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs, insert_num);

    auto values_hps = get_values_hps(insert_num, values_bytes_indexs, values_bytes);
    int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
    int keys_indexs_size = util::indexs_size_sum(insert_num);
    int64_t values_bytes_size =
        util::elements_size_sum(values_bytes_indexs, insert_num);
    int values_indexs_size = util::indexs_size_sum(insert_num);
    int values_hps_size = insert_num;
    std::vector<std::string> columns = {"method", "data_num", "throughput"};

    exp_util::CSVDataRecorder insert_recorder(columns, "./data/insert_ycsb_thread.csv");

    {
        CHECK_ERROR(cudaDeviceReset());
        CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
        CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
        CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
        CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
        CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
        GPUHashMultiThread::load_constants();
        GpuMPT::Compress::MPT gpu_mpt_olc;
        auto [d_hash_nodes, hash_nodes_num] =
            gpu_mpt_olc.puts_latching_with_valuehp_v2_with_record(
                keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
                values_hps, insert_num, insert_recorder, insert_num);
        gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
        auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
        printf("GPU olc hash is: ");
        cutil::println_hex(hash, hash_size);
    }

    {
        CHECK_ERROR(cudaDeviceReset());
        CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
        CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
        CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
        CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
        CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
        GPUHashMultiThread::load_constants();
        GpuMPT::Compress::MPT gpu_mpt_two;
        auto [d_hash_nodes, hash_nodes_num] =
            gpu_mpt_two.puts_2phase_with_valuehp_with_recorder(
                keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
                values_hps, insert_num, insert_recorder, insert_num);
        gpu_mpt_two.hash_onepass_v2(d_hash_nodes, hash_nodes_num);

        auto [hash, hash_size] = gpu_mpt_two.get_root_hash();
        printf("GPU two hash is: ");
        cutil::println_hex(hash, hash_size);
    }
}

TEST(ethtxn, read_ethaddress_data_all)
{
    std::vector<std::pair<std::string, std::string>> kvs;
    std::vector<int> txns_addresses_indexs;
    bench::ethtxn::read_ethaddress_data_all(bench::ethtxn::ETHADDRESS_PATH, kvs);
    std::cout << "kvs: " << kvs.size() << std::endl;
    // sort kvs
    std::sort(kvs.begin(), kvs.end(), [](const std::pair<std::string, std::string> &a, const std::pair<std::string, std::string> &b)
              { return a.first < b.first; });

    // write kvs to a csv file, each line one key-value pair
    std::ofstream out("ethaddress.csv");
    for (auto &kv : kvs)
    {
        out << kv.first << "," << kv.second << std::endl;
    }
    out.close();
}