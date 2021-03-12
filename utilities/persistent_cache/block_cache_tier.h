//  Copyright (c) 2013, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
#pragma once

#ifndef ROCKSDB_LITE

#ifndef  OS_WIN
#include <unistd.h>
#endif // ! OS_WIN

#include <atomic>
#include <list>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <fcntl.h>

#include "rocksdb/cache.h"
#include "rocksdb/comparator.h"
#include "rocksdb/persistent_cache.h"

#include "utilities/persistent_cache/block_cache_tier_file.h"
#include "utilities/persistent_cache/block_cache_tier_metadata.h"
#include "utilities/persistent_cache/persistent_cache_util.h"

#include "memory/arena.h"
#include "memtable/skiplist.h"
#include "monitoring/histogram.h"
#include "port/port.h"
#include "util/coding.h"
#include "util/crc32c.h"
#include "util/mutexlock.h"

namespace rocksdb {

//
// Block cache tier implementation
//
class BlockCacheTier : public PersistentCacheTier {
 public:
  explicit BlockCacheTier(const PersistentCacheConfig& opt)
      : opt_(opt),
        insert_ops_(static_cast<size_t>(opt_.max_write_pipeline_backlog_size)),
        buffer_allocator_(opt.write_buffer_size, opt.write_buffer_count()),
        writer_(this, opt_.writer_qdepth, static_cast<size_t>(opt_.writer_dispatch_size)) {
    Info(opt_.log, "Initializing allocator. size=%d B count=%" ROCKSDB_PRIszt,
         opt_.write_buffer_size, opt_.write_buffer_count());
  }

  virtual ~BlockCacheTier() {
    // Close is re-entrant so we can call close even if it is already closed
    Close();
    assert(!insert_th_.joinable());
  }

  Status Insert(const Slice& key, const char* data, const size_t size,std::string fanme = "") override;
  Status Lookup(const Slice& key, std::unique_ptr<char[]>* data,
                size_t* size,std::string fanme = "") override;
  Status Open() override;
  Status Close() override;
  bool Erase(const Slice& key) override;
  bool Reserve(const size_t size) override;

  bool IsCompressed() override { return opt_.is_compressed; }

  std::string GetPrintableOptions() const override { return opt_.ToString(); }

  PersistentCache::StatsType Stats() override;

  void TEST_Flush() override {
    while (insert_ops_.Size()) {
      /* sleep override */
      Env::Default()->SleepForMicroseconds(1000000);
    }
  }

 private:
  // Percentage of cache to be evicted when the cache is full
  static const size_t kEvictPct = 10;
  // Max attempts to insert key, value to cache in pipelined mode
  static const size_t kMaxRetry = 3;

  // Pipelined operation
  struct InsertOp {
    explicit InsertOp(const bool signal) : signal_(signal) {}
    explicit InsertOp(std::string&& key, const std::string& data)
        : key_(std::move(key)), data_(data) {}
    ~InsertOp() {}

    InsertOp() = delete;
    InsertOp(InsertOp&& /*rhs*/) = default;
    InsertOp& operator=(InsertOp&& rhs) = default;

    // used for estimating size by bounded queue
    size_t Size() { return data_.size() + key_.size(); }

    std::string key_;
    std::string data_;
    bool signal_ = false;  // signal to request processing thread to exit
  };

  // entry point for insert thread
  void InsertMain();
  // insert implementation
  Status InsertImpl(const Slice& key, const Slice& data);
  // Create a new cache file
  Status NewCacheFile();
  // Get cache directory path
  std::string GetCachePath() const { return opt_.path + "/cache"; }
  // Cleanup folder
  Status CleanupCacheFolder(const std::string& folder);

  // Statistics
  struct Statistics {
    HistogramImpl bytes_pipelined_;
    HistogramImpl bytes_written_;
    HistogramImpl bytes_read_;
    HistogramImpl read_hit_latency_;
    HistogramImpl read_miss_latency_;
    HistogramImpl write_latency_;
    std::atomic<uint64_t> cache_hits_{0};
    std::atomic<uint64_t> cache_misses_{0};
    std::atomic<uint64_t> cache_errors_{0};
    std::atomic<uint64_t> insert_dropped_{0};

    double CacheHitPct() const {
      const auto lookups = cache_hits_ + cache_misses_;
      return lookups ? 100 * cache_hits_ / static_cast<double>(lookups) : 0.0;
    }

    double CacheMissPct() const {
      const auto lookups = cache_hits_ + cache_misses_;
      return lookups ? 100 * cache_misses_ / static_cast<double>(lookups) : 0.0;
    }
  };

  port::RWMutex lock_;                          // Synchronization
  const PersistentCacheConfig opt_;             // BlockCache options
  BoundedQueue<InsertOp> insert_ops_;           // Ops waiting for insert
  rocksdb::port::Thread insert_th_;                       // Insert thread
  uint32_t writer_cache_id_ = 0;                // Current cache file identifier
  WriteableCacheFile* cache_file_ = nullptr;    // Current cache file reference
  CacheWriteBufferAllocator buffer_allocator_;  // Buffer provider
  ThreadedWriter writer_;                       // Writer threads
  BlockCacheTierMetadata metadata_;             // Cache meta data manager
  std::atomic<uint64_t> size_{0};               // Size of the cache
  Statistics stats_;                                 // Statistics
};



//wp


#define SST_SIZE (16 * 1024*1024)  //单个SST所占空间 800KB
#define SPACE_SIZE (4 * 1024)  //单个空间大小     4KB

struct Record  // KV记录结构
{
  std::vector<int> offset;
  size_t size;
};
struct DLinkedNode  //双向链表节点
{
  std::string key;
  Record value;
  DLinkedNode* prev;
  DLinkedNode* next;
  DLinkedNode() : prev(nullptr), next(nullptr) {}
};

class SST_space  // cache 管理单个SST所占空间
{
 public:
  SST_space(){};
  void Set_Par(int fd_, uint32_t num, uint32_t begin_) {
    fd = fd_;
    begin = begin_;
    all_num = num;
    empty_num = num;
    bit_map.resize(num);
    bit_map.assign(num, 0);
    head = new DLinkedNode();
    tail = new DLinkedNode();
    head->next = tail;
    tail->prev = head;
  }
  SST_space(int fd_, int num, int begin_)
      : fd(fd_), begin(begin_), all_num(num), empty_num(num)
      {
    bit_map.resize(num);
    bit_map.assign(num, 0);
    head = new DLinkedNode();
    tail = new DLinkedNode();
    head->next = tail;
    tail->prev = head;
  }
  virtual ~SST_space() {
    delete head;
    delete tail;
  }

  std::string Get(std::string key);
  Status Get(const std::string key, std::unique_ptr<char[]>* data,
             size_t* size);

  void Put(const std::string &key, const std::string &value);

  Status Put(const std::string key, const char* data, const size_t size);

 private:
  void removeRecord(Record* record) {
    int free_num = record->offset.size();
    for (uint32_t i = 0; i < record->offset.size(); i++) {
      int index = record->offset[i] / SPACE_SIZE;
      bit_map[index] = 0;
    }
    record->offset.clear();
    empty_num += free_num;
  }
  void addToHead(DLinkedNode* node) {
    node->prev = head;
    node->next = head->next;
    head->next->prev = node;
    head->next = node;
  }

  void removeNode(DLinkedNode* node) {
    node->prev->next = node->next;
    node->next->prev = node->prev;
  }

  void moveToHead(DLinkedNode* node) {
    removeNode(node);
    addToHead(node);
  }
  DLinkedNode* removeTail() {
    DLinkedNode* node = tail->prev;
    removeNode(node);
    return node;
  }

 private:
  port::Mutex lock;
  int fd=-1;
  uint32_t begin;             //指向该SST空间起始位置
  std::vector<bool> bit_map;  // bitmap暂时用bool数组代替
  uint32_t all_num;           //总空间数
  uint32_t empty_num;         //空空间数
  std::unordered_map<std::string, DLinkedNode*> cache;
  DLinkedNode *head, *tail;
};

class myCache : public PersistentCacheTier {
 public:
  explicit myCache(const PersistentCacheConfig& opt) : opt_(opt) {}
  virtual ~myCache(){
      Close();
  }

 private:
  // Pipelined operation
  struct myInsertOp {
    explicit myInsertOp(const bool signal) :signal_(signal) {}
    explicit myInsertOp(std::string&& key, const std::string& value, 
                        const std::string& fname)
        : key_(std::move(key)), value_(value), fname_(fname) {}
    ~myInsertOp() {}

    myInsertOp() = delete;
    myInsertOp(myInsertOp&& /*rhs*/) = default;
    myInsertOp& operator=(myInsertOp&& rhs) = default;

    // used for estimating size by bounded queue
    size_t Size() { return value_.size() + key_.size(); }


    std::string key_;
    std::string value_;
    std::string fname_;
    bool signal_ = false;  // signal to request processing thread to exit
  };

  int getIndex(
      std::string fname)  // filename 格式一般为 /.../0000123.sst
                          // 此处使用sst序号作为index，若非该格式 则放入最后
  {
    if (fname.size() == 0) {
      return 0;
    }
    int i = fname.size() - 1;
    while (fname[i] != '.') {
      i--;
    }
    i--;
    int j=i;
    int sum = 0;
    while (fname[i] >= '0' && fname[i] <= '9' && i >= 0) {

      i--;
    }
    i++;
    while(fname[i]=='0')
    {
      i++;
    }
    while(j>=i)
    {
            sum = sum * 10 + fname[i] - '0';
            i++;
    }
    // if(stat)
    // {
    //   fprintf(fp,"%d\n",sum);
    // }
    
    return sum % NUM;
  }

 public:
  void InsertMain();
  Status Insert(const Slice& key, const char* data, const size_t size,
                std::string fanme = "") override;
  Status InsertImpl(const Slice& key, const char* data, const size_t size,
                std::string fanme = "");

  Status Lookup(const Slice& key, std::unique_ptr<char[]>* data, size_t* size,
                std::string fanme = "") override;

  Status Insert2(const std::string& key, const std::string& value,
                        std::string &fname) ;


  Status Open() override;
  Status Close() override;
  bool Erase(const Slice& key) override;
  bool Reserve(const size_t size) override;

  bool IsCompressed() override;

  std::string GetPrintableOptions() const override;

  // PersistentCache::StatsType Stats() override;
 private:
  BoundedQueue<myInsertOp> insert_ops_;  // Ops waiting for insert
  rocksdb::port::Thread insert_th_;      // Insert thread
  //port::Mutex lock_;                   // Synchronization

  int fd=-1;
  int NUM;

  const PersistentCacheConfig opt_;  // BlockCache options

  //std::vector<SST_space> v;
  SST_space v[260];
};




}  // namespace rocksdb

#endif
