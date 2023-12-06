// #define GETH 1
#define LEDGERDB 1

#include <unistd.h>

#include <mutex>
#include <tuple>

#include "libgmpt.h"
#include "mpt/cpu_mpt.cuh"
#include "mpt/gpu_mpt.cuh"
#include "mpt/node.cuh"
#include "util/experiments.cuh"
#include "util/utils.cuh"

struct Tries {
  GpuMPT::Compress::MPT *state_trie;
  GpuMPT::Compress::MPT *transaction_trie;
  GpuMPT::Compress::MPT *receipt_trie;
};

static struct Tries *tries = nullptr;

struct ThreadSafeMPT {
  GpuMPT::Compress::MPT mpt;
  std::mutex mu;
  uint8_t hash[HASH_SIZE];
  int hash_size;
};

// TODO: Redefine ALLOCATOR
int preprocess() {
  printf("preprocess()\n");
  CHECK_ERROR(cudaSetDevice(1));
  CHECK_ERROR(cudaDeviceReset());
  if (tries != nullptr) {
    // delete (GpuMPT::Compress::MPT *)tries->state_trie;
    // delete (GpuMPT::Compress::MPT *)tries->receipt_trie;
    // delete (GpuMPT::Compress::MPT *)tries->transaction_trie;
    delete tries;
  }
  GPUHashMultiThread::load_constants();
  tries = new Tries{};
  tries->state_trie = new GpuMPT::Compress::MPT{};
  tries->receipt_trie = new GpuMPT::Compress::MPT{};
  tries->transaction_trie = new GpuMPT::Compress::MPT{};
  printf("state_trie %p, receipt_trie %p, transaction_trie %p\n",
         tries->state_trie, tries->receipt_trie, tries->transaction_trie);
  return 1;
}

const uint8_t *build_mpt_2phase(enum TrieType trie_type,
                                const uint8_t *keys_hexs, int *keys_hexs_indexs,
                                const uint8_t *values_bytes,
                                int64_t *values_bytes_indexs,
                                const uint8_t **values_hps, int insert_num) {
  CHECK_ERROR(cudaSetDevice(1));
  assert(tries != nullptr);
  GpuMPT::Compress::MPT *mpt = nullptr;
  if (trie_type == TrieType::RECEIPT_TRIE) {
    mpt = tries->receipt_trie;
  } else if (trie_type == TrieType::STATE_TRIE) {
    mpt = tries->state_trie;
  } else if (trie_type == TrieType::TRANSACTION_TRIE) {
    mpt = tries->transaction_trie;
  }
  assert(mpt != nullptr);

  // choose the second GPU
  perf::CpuMultiTimer<perf::us> timer;
  timer.start();
  if (values_hps == nullptr) {
    values_hps = new const uint8_t *[insert_num];
    for (int i = 0; i < insert_num; i++) {
      values_hps[i] = nullptr;
    }
  }
  timer.stop();
  // calculate size to pre-pin

  perf::CpuTimer<perf::us> timer_pin;
  timer_pin.start();
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  // int values_hps_size = insert_num;

  // printf("keys_hexs_size: %d\n", keys_hexs_size);
  // printf("keys_indexs_size: %d\n", keys_indexs_size);
  // printf("values_bytes_size: %ld\n", values_bytes_size);
  // printf("values_indexs_size: %d\n", values_indexs_size);
  // printf("values_hps_size: %d\n", values_hps_size);
  timer_pin.stop();

  // printf("pre-pin mmeory: %dus\n", timer_pin.get());

  // TODO
  // for 100000: insert is the same w/o pinhost
  // w/ pinhost: load_constants + init is faster
  // w/o pinhost: load_constants + init is slower
  // CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
  // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
  // CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
  // CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
  timer.stop();

  // GPUHashMultiThread::load_constants();
  GpuMPT::Compress::MPT &gpu_mpt_olc = *mpt;
  timer.stop();

  auto [d_hash_nodes, hash_nodes_num] =
      gpu_mpt_olc.puts_latching_with_valuehp_v2(
          keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
          values_hps, insert_num);
  gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);

  auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
  timer.stop();
  // printf("GPU olc hash is: ");
  cutil::println_hex(hash, hash_size);

  // CHECK_ERROR(cudaDeviceReset());
  timer.stop();

  // printf(
  //     "\t[Timer] valuehps: %dus\n"
  //     "\t[Timer] pre-pin mmeory: %dus\n"
  //     "\t[Timer] ld constant & init: %dus\n"
  //     "\t[Timer] insert: %dus\n"
  //     "\t[Timer] reset: %dus\n",
  //     timer.get(0), timer.get(1), timer.get(2), timer.get(3), timer.get(4));
  return hash;
}

const uint8_t *build_mpt_olc(enum TrieType trie_type, const uint8_t *keys_hexs,
                             int *keys_hexs_indexs, const uint8_t *values_bytes,
                             int64_t *values_bytes_indexs,
                             const uint8_t **values_hps, int insert_num) {
  CHECK_ERROR(cudaSetDevice(1));
  assert(tries != nullptr);
  GpuMPT::Compress::MPT *mpt = nullptr;
  if (trie_type == TrieType::RECEIPT_TRIE) {
    mpt = tries->receipt_trie;
  } else if (trie_type == TrieType::STATE_TRIE) {
    mpt = tries->state_trie;
  } else if (trie_type == TrieType::TRANSACTION_TRIE) {
    mpt = tries->transaction_trie;
  }
  assert(mpt != nullptr);

  // choose the second GPU
  perf::CpuMultiTimer<perf::us> timer;
  timer.start();
  if (values_hps == nullptr) {
    values_hps = new const uint8_t *[insert_num];
    for (int i = 0; i < insert_num; i++) {
      values_hps[i] = nullptr;
    }
  }
  timer.stop();
  // calculate size to pre-pin

  perf::CpuTimer<perf::us> timer_pin;
  timer_pin.start();
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  // int values_hps_size = insert_num;

  // printf("keys_hexs_size: %d\n", keys_hexs_size);
  // printf("keys_indexs_size: %d\n", keys_indexs_size);
  // printf("values_bytes_size: %ld\n", values_bytes_size);
  // printf("values_indexs_size: %d\n", values_indexs_size);
  // printf("values_hps_size: %d\n", values_hps_size);
  timer_pin.stop();

  // printf("pre-pin mmeory: %dus\n", timer_pin.get());

  // TODO
  // for 100000: insert is the same w/o pinhost
  // w/ pinhost: load_constants + init is faster
  // w/o pinhost: load_constants + init is slower
  // CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
  // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
  // CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
  // CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
  timer.stop();

  // GPUHashMultiThread::load_constants();
  GpuMPT::Compress::MPT &gpu_mpt_olc = *mpt;
  timer.stop();

  auto [d_hash_nodes, hash_nodes_num] =
      gpu_mpt_olc.puts_latching_with_valuehp_v2(
          keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
          values_hps, insert_num);
  gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);

  auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();
  timer.stop();
  // printf("GPU olc hash is: ");
  cutil::println_hex(hash, hash_size);

  // CHECK_ERROR(cudaDeviceReset());
  timer.stop();

  // printf(
  //     "\t[Timer] valuehps: %dus\n"
  //     "\t[Timer] pre-pin mmeory: %dus\n"
  //     "\t[Timer] ld constant & init: %dus\n"
  //     "\t[Timer] insert: %dus\n"
  //     "\t[Timer] reset: %dus\n",
  //     timer.get(0), timer.get(1), timer.get(2), timer.get(3), timer.get(4));
  return hash;
}

struct nodeset get_all_nodes(enum TrieType trie_type, const uint8_t *keys_hexs,
                             int *keys_hexs_indexs, int num) {
  printf("get_all_nodes\n");
  CHECK_ERROR(cudaSetDevice(1));

  assert(tries != nullptr && trie_type == TrieType::STATE_TRIE);
  GpuMPT::Compress::MPT *mpt = tries->state_trie;
  delete tries->receipt_trie;
  delete tries->transaction_trie;
  // if (trie_type == TrieType::RECEIPT_TRIE) {
  //   mpt = tries->receipt_trie;
  // } else if (trie_type == TrieType::STATE_TRIE) {
  //   mpt = tries->state_trie;
  // } else if (trie_type == TrieType::TRANSACTION_TRIE) {
  //   mpt = tries->transaction_trie;
  // }
  assert(mpt != nullptr);
  struct nodeset set {};

  mpt->flush_dirty_nodes(keys_hexs, keys_hexs_indexs, num, set.hashs, set.encs,
                         set.encs_indexs, set.num);
  return set;
}

//
const uint8_t *insert_mpt_2phase(void *mpt, const uint8_t *keys_hexs,
                                 int *keys_hexs_indexs,
                                 const uint8_t *values_bytes,
                                 int64_t *values_bytes_indexs,
                                 const uint8_t **values_hps, int insert_num) {
  assert(mpt != nullptr);

  perf::CpuMultiTimer<perf::us> timer;
  timer.start();
  if (values_hps == nullptr) {
    values_hps = new const uint8_t *[insert_num];
    for (int i = 0; i < insert_num; i++) {
      values_hps[i] = nullptr;
    }
  }
  timer.stop();
  // calculate size to pre-pin

  // perf::CpuTimer<perf::us> timer_pin;
  // timer_pin.start();
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  // int values_hps_size = insert_num;

  // printf("keys_hexs_size: %d\n", keys_hexs_size);
  // printf("keys_indexs_size: %d\n", keys_indexs_size);
  // printf("values_bytes_size: %ld\n", values_bytes_size);
  // printf("values_indexs_size: %d\n", values_indexs_size);
  // printf("values_hps_size: %d\n", values_hps_size);
  // timer_pin.stop();

  // printf("pre-pin mmeory: %dus\n", timer_pin.get());

  // TODO
  // for 100000: insert is the same w/o pinhost
  // w/ pinhost: load_constants + init is faster
  // w/o pinhost: load_constants + init is slower
  // CHECK_ERROR(gutil::PinHost(keys_hexs, keys_hexs_size));
  // CHECK_ERROR(gutil::PinHost(keys_hexs_indexs, keys_indexs_size));
  // CHECK_ERROR(gutil::PinHost(values_bytes, values_bytes_size));
  // CHECK_ERROR(gutil::PinHost(values_bytes_indexs, values_indexs_size));
  // CHECK_ERROR(gutil::PinHost(values_hps, values_hps_size));
  // timer.stop();

  // GPUHashMultiThread::load_constants();
  ThreadSafeMPT &thread_safe_mpt = *(ThreadSafeMPT *)mpt;

  {
    std::lock_guard<std::mutex> lck(thread_safe_mpt.mu);
    GpuMPT::Compress::MPT &gpu_mpt_olc = thread_safe_mpt.mpt;
    // timer.stop();

    // TODO: free d_keys_hexs_indexs
    // TODO: free d_value_indexs
    auto [d_hash_nodes, hash_nodes_num] = gpu_mpt_olc.puts_2phase_with_valuehp(
        keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
        values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);
    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();

    assert(hash_size == HASH_SIZE);
    memcpy(thread_safe_mpt.hash, hash, hash_size);
    thread_safe_mpt.hash_size = hash_size;

    // timer.stop();
    // printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);

    // CHECK_ERROR(cudaDeviceReset());
    // timer.stop();

    // printf(
    //     "\t[Timer] valuehps: %dus\n"
    //     "\t[Timer] pre-pin mmeory: %dus\n"
    //     "\t[Timer] ld constant & init: %dus\n"
    //     "\t[Timer] insert: %dus\n"
    //     "\t[Timer] reset: %dus\n",
    //     timer.get(0), timer.get(1), timer.get(2), timer.get(3),
    //     timer.get(4));
    return hash;
  }
}

const uint8_t *insert_mpt_olc(void *mpt, const uint8_t *keys_hexs,
                              int *keys_hexs_indexs,
                              const uint8_t *values_bytes,
                              int64_t *values_bytes_indexs,
                              const uint8_t **values_hps, int insert_num) {
  assert(mpt != nullptr);

  perf::CpuMultiTimer<perf::us> timer;
  timer.start();
  if (values_hps == nullptr) {
    values_hps = new const uint8_t *[insert_num];
    for (int i = 0; i < insert_num; i++) {
      values_hps[i] = nullptr;
    }
  }
  timer.stop();
  // calculate size to pre-pin

  // perf::CpuTimer<perf::us> timer_pin;
  // timer_pin.start();
  int keys_hexs_size = util::elements_size_sum(keys_hexs_indexs, insert_num);
  int keys_indexs_size = util::indexs_size_sum(insert_num);
  int64_t values_bytes_size =
      util::elements_size_sum(values_bytes_indexs, insert_num);
  int values_indexs_size = util::indexs_size_sum(insert_num);
  // int values_hps_size = insert_num;

  // printf("keys_hexs_size: %d\n", keys_hexs_size);
  // printf("keys_indexs_size: %d\n", keys_indexs_size);
  // printf("values_bytes_size: %ld\n", values_bytes_size);
  // printf("values_indexs_size: %d\n", values_indexs_size);
  // printf("values_hps_size: %d\n", values_hps_size);
  // timer_pin.stop();

  // printf("pre-pin mmeory: %dus\n", timer_pin.get());

  ThreadSafeMPT &thread_safe_mpt = *(ThreadSafeMPT *)mpt;

  {
    std::lock_guard<std::mutex> lck(thread_safe_mpt.mu);
    GpuMPT::Compress::MPT &gpu_mpt_olc = thread_safe_mpt.mpt;

    auto [d_hash_nodes, hash_nodes_num] =
        gpu_mpt_olc.puts_latching_with_valuehp_v2(
            keys_hexs, keys_hexs_indexs, values_bytes, values_bytes_indexs,
            values_hps, insert_num);
    gpu_mpt_olc.hash_onepass_v2(d_hash_nodes, hash_nodes_num);

    auto [hash, hash_size] = gpu_mpt_olc.get_root_hash();

    assert(hash_size == HASH_SIZE);
    memcpy(thread_safe_mpt.hash, hash, hash_size);
    thread_safe_mpt.hash_size = hash_size;
    // timer.stop();
    // printf("GPU olc hash is: ");
    cutil::println_hex(hash, hash_size);

    // CHECK_ERROR(cudaDeviceReset());
    // timer.stop();

    // printf(
    //     "\t[Timer] valuehps: %dus\n"
    //     "\t[Timer] pre-pin mmeory: %dus\n"
    //     "\t[Timer] ld constant & init: %dus\n"
    //     "\t[Timer] insert: %dus\n"
    //     "\t[Timer] reset: %dus\n",
    //     timer.get(0), timer.get(1), timer.get(2), timer.get(3),
    //     timer.get(4));
    return hash;
  }
}

///
/// get value and proofs
void get_proofs(  //
    void *mpt, const uint8_t *keys_hexs, int *keys_hexs_indexs, int get_num,
    const uint8_t **&values_hps_get, const int *&values_sizes_get,  // values
    const uint8_t *&proofs, const int *&proofs_indexs, const uint8_t *&hash,
    int &hash_size) {  // proofs
  assert(!values_hps_get && !values_sizes_get && !proofs && !proofs_indexs);
  ThreadSafeMPT &thread_safe_mpt = *(ThreadSafeMPT *)mpt;
  {
    std::lock_guard<std::mutex> lck(thread_safe_mpt.mu);
    GpuMPT::Compress::MPT &gpu_mpt_olc = thread_safe_mpt.mpt;

    // perf::CpuTimer<perf::us> timer;
    // timer.start();

    uint8_t *m_proofs = nullptr;
    int *m_proofs_indexs = nullptr;
    auto m_values_hps_get = new const uint8_t *[get_num];
    auto m_values_sizes_get = new int[get_num];

    gpu_mpt_olc.get_proofs(keys_hexs, keys_hexs_indexs, get_num,
                           m_values_hps_get, m_values_sizes_get, m_proofs,
                           m_proofs_indexs);

    values_hps_get = m_values_hps_get, values_sizes_get = m_values_sizes_get;
    proofs = m_proofs, proofs_indexs = m_proofs_indexs;

    assert(thread_safe_mpt.hash_size == HASH_SIZE);
    uint8_t *hash_ = new uint8_t[thread_safe_mpt.hash_size];
    memcpy(hash_, thread_safe_mpt.hash, thread_safe_mpt.hash_size);

    hash = hash_, hash_size = thread_safe_mpt.hash_size;

    // timer.stop();
    // printf("get_proofs time inside the lock section is %d\n", timer.get());
  }
}

bool verify_proof_single(const uint8_t *key_hex, int key_hex_size,
                         const uint8_t *digest, int digest_size,
                         const uint8_t *value, int value_size,
                         const uint8_t *proof, int proof_size) {
  // TODO
  return GpuMPT::Compress::MPT::verify_proof_cpu(key_hex, key_hex_size, digest,
                                                 digest_size, value, value_size,
                                                 proof, proof_size);
}

void *init_mpt() {
  // assert(tries == nullptr);
  GPUHashMultiThread::load_constants();
  // return new GpuMPT::Compress::MPT{};
  return new ThreadSafeMPT;
}

void free_mpt(void *mpt) {
  // TODO: currently free all
  CHECK_ERROR(cudaDeviceReset());
}