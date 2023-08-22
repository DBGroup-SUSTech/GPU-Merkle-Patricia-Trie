#pragma once
#include "hash/gpu_hash_kernel.cuh"
#include "util/timer.cuh"

namespace exp_util {
class ProfileResult {};

/// @brief init env when construct, clear env when destruct
template <typename Timer>
class Profiler {
 public:
  Profiler() {}
  // virtual ~Profiler() {}

  virtual void start() { timer_.start(); }
  virtual void stop() { timer_.stop(); }
  virtual void print() { printf(" %d us\n", timer_.get()); }

 public:
  Timer timer_;
};

template <typename Timer>
class LookupProfiler : public Profiler<Timer> {
 public:
  static_assert(std::is_same_v<Timer, perf::CpuTimer<perf::us>>);

  LookupProfiler(const char* competitor, int lookup_num, int record_num)
      : competitor_(competitor),
        lookup_num_(lookup_num),
        record_num_(record_num) {}
  // virtual ~LookupProfiler() {}

  void print() override {
    printf(
        "%s lookup response time: %d us for %d operations and trie with %d "
        "records\n",
        competitor_, Profiler<Timer>::timer_.get(), lookup_num_, record_num_);
  }

 private:
  const char* competitor_;
  int lookup_num_;
  int record_num_;
};

template <typename Timer>
class InsertProfiler : public Profiler<Timer> {
 public:
  static_assert(std::is_same_v<Timer, perf::CpuTimer<perf::us>>);

  InsertProfiler(const char* competitor, int insert_num, int record_num)
      : competitor_(competitor),
        insert_num_(insert_num),
        record_num_(record_num) {}
  // virtual ~LookupProfiler() {}

  void print() override {
    int qps = insert_num_ * 1000.0 / Profiler<Timer>::timer_.get() * 1000.0;
    printf(
        "%s Insert throughput: %d qps for %d operations and trie with %d "
        "records\n",
        competitor_, qps, insert_num_, record_num_);
  }

  std::string get_throughput() {
    int qps = insert_num_ * 1000.0 / Profiler<Timer>::timer_.get() * 1000.0;
    return std::to_string(qps);
  }

  std::string get_competitor() { return std::string(competitor_); }

 public:
  const char* competitor_;
  int insert_num_;
  int record_num_;
};

template <typename Timer>
class HashProfiler : public Profiler<Timer> {
 public:
  static_assert(std::is_same_v<Timer, perf::CpuTimer<perf::us>>);
  HashProfiler(const char* competitor, int insert_num, int record_num)
      : competitor_(competitor),
        insert_num_(insert_num),
        record_num_(record_num) {}
  // virtual ~LookupProfiler() {}

  void print() override {
    int qps = insert_num_ * 1000.0 / Profiler<Timer>::timer_.get() * 1000.0;
    printf(
        "%s Hash throughput: %d qps for %d operations and trie with %d "
        "records\n",
        competitor_, qps, insert_num_, record_num_);
  }

 private:
  const char* competitor_;
  int insert_num_;
  int record_num_;
};

class CSVDataRecorder {
 public:
  CSVDataRecorder() {}
  CSVDataRecorder(std::vector<std::string> column_names, std::string file_name)
      : column_names_(column_names), file_name_(file_name) {}
  ~CSVDataRecorder() {}
  void update_row(std::vector<std::string> row) {
    for (int i = 0; i < column_names_.size(); i++) {
      data_[column_names_[i]].push_back(row[i]);
    }
  }
  void persist_data() {
    // check if file exists
    std::ifstream f(file_name_);
    bool file_exists = f.good();
    if (!file_exists) {
      std::ofstream file(file_name_);
      for (int i = 0; i < column_names_.size(); i++) {
        file << column_names_[i] << ",";
      }
      file << "\n";
      file.close();
    } 
    std::ofstream file(file_name_, std::ios_base::app);
    for (int i = 0; i < data_[column_names_[0]].size(); i++) {
      for (int j = 0; j < column_names_.size(); j++) {
        file << data_[column_names_[j]][i] << ",";
      }
      file << "\n";
    }
    file.close();
  }
  public:
  //column_names
  std::vector<std::string> column_names_;
  std::unordered_map<std::string, std::vector<std::string>> data_;
  std::string file_name_;
};

}  // namespace exp_util