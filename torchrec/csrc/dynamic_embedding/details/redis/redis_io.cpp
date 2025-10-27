/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torchrec/csrc/dynamic_embedding/details/redis/redis_io.h>
#include <torchrec/csrc/dynamic_embedding/details/redis/url.h>
#include <iostream>
#include <variant>

namespace torchrec::redis {

int parse_integer(std::string_view param_str, std::string_view param_key) {
  return std::stoi(std::string(param_str.substr(param_key.size())));
}

uint32_t parse_duration(
    std::string_view param_str,
    std::string_view param_key) {
  auto param_value = param_str.substr(param_key.size());
  if (param_value.empty()) {
    throw std::invalid_argument("no value for " + std::string(param_str));
  }
  double duration;
  if (param_value.ends_with("ms")) {
    duration =
        std::stod(std::string(param_value.substr(0, param_value.size() - 2)));
  } else if (param_value.ends_with("s")) {
    duration =
        std::stod(std::string(param_value.substr(0, param_value.size() - 1))) *
        1000;
  } else if (param_value.ends_with("m")) {
    duration =
        std::stod(std::string(param_value.substr(0, param_value.size() - 1))) *
        1000 * 60;
  } else {
    throw std::invalid_argument(
        "no supported time unit (ms, s, m) in " + std::string(param_str));
  }
  return static_cast<uint32_t>(duration);
}

Option parse_option(std::string_view config_str) {
  Option option;
  url_parser::Url url = url_parser::parse_url(config_str);

  if (url.authority.has_value()) {
    option.username = std::move(url.authority->username);
    option.password = std::move(url.authority->password);
  }

  option.host = std::move(url.host);

  if (url.port.has_value()) {
    option.port = url.port.value();
  }

  if (url.param.has_value()) {
    std::string_view param_str = url.param.value();
    while (!param_str.empty()) {
      auto and_pos = param_str.find("&&");
      std::string_view single_param_str;
      if (and_pos != std::string_view::npos) {
        single_param_str = param_str.substr(0, and_pos);
        param_str = param_str.substr(and_pos + 2);
      } else {
        single_param_str = param_str;
        param_str = "";
      }

      if (single_param_str.starts_with("num_threads=")) {
        option.num_io_threads = parse_integer(single_param_str, "num_threads=");
      } else if (single_param_str.starts_with("db=")) {
        option.db = parse_integer(single_param_str, "db=");
      } else if (single_param_str.starts_with("prefix=")) {
        option.prefix = single_param_str.substr(std::string("prefix=").size());
      } else if (single_param_str.starts_with("timeout=")) {
        option.timeout_ms = parse_duration(single_param_str, "timeout=");
      } else if (single_param_str.starts_with("heartbeat=")) {
        option.heart_beat_interval_ms =
            parse_duration(single_param_str, "heartbeat=");
      } else if (single_param_str.starts_with("retry_limit=")) {
        option.retry_limit = parse_integer(single_param_str, "retry_limit=");
      } else if (single_param_str.starts_with("chunk_size=")) {
        option.chunk_size = parse_integer(single_param_str, "chunk_size=");
      } else {
        throw std::invalid_argument(
            "unknown parameter: " + std::string(single_param_str));
      }
    }
  }

  return option;
}

Redis::Redis(Option opt) : opt_(std::move(opt)) {
  TORCH_CHECK(opt_.num_io_threads != 0, "num_io_threads must not be empty");
  TORCH_CHECK(
      opt_.heart_beat_interval_ms != 0,
      "heart beat interval must not be zero.");
  for (size_t i = 0; i < opt_.num_io_threads; ++i) {
    start_thread();
  }
}

void Redis::start_thread() {
  auto connection = connect();
  heartbeat(connection);

  io_threads_.emplace_back(
      [connection = std::move(connection), this]() mutable {
        std::chrono::milliseconds heart_beat(opt_.heart_beat_interval_ms);
        while (true) {
          std::function<void(helper::ContextPtr&)> todo;
          bool heartbeat_timeout;
          {
            std::unique_lock<std::mutex> lock(this->jobs_mutex_);
            heartbeat_timeout = !jobs_not_empty_.wait_for(
                lock, heart_beat, [this] { return !jobs_.empty(); });
            if (!heartbeat_timeout) {
              todo = std::move(jobs_.front());
              jobs_.pop_front();
            }
          }

          if (heartbeat_timeout) {
            heartbeat(connection);
            continue;
          }

          if (!todo) {
            break;
          }
          todo(connection);
        }
      });
}

void Redis::heartbeat(helper::ContextPtr& connection) {
  for (uint32_t retry = 0; retry < opt_.retry_limit; ++retry) {
    try {
      auto reply = helper::ReplyPtr(
          reinterpret_cast<redisReply*>(
              redisCommand(connection.get(), "PING")));
      TORCH_CHECK(
          reply && reply->type == REDIS_REPLY_STRING,
          "Ping should return string");
      auto rsp = std::string_view(reply->str, reply->len);
      TORCH_CHECK(rsp == "PONG", "ping/pong error");
    } catch (...) {
      // reconnect if heart beat error
      connection = connect();
    }
  }
}

helper::ContextPtr Redis::connect() const {
  helper::ContextPtr connection;
  if (opt_.timeout_ms == 0) {
    connection = helper::ContextPtr(redisConnect(opt_.host.c_str(), opt_.port));
  } else {
    struct timeval interval{};
    interval.tv_sec = opt_.timeout_ms / 1000;
    interval.tv_usec = opt_.timeout_ms % 1000 * 1000;
    connection = helper::ContextPtr(
        redisConnectWithTimeout(opt_.host.c_str(), opt_.port, interval));
  }
  TORCH_CHECK(
      !connection->err,
      "connect to %s:%d error occurred %s",
      opt_.host,
      opt_.port,
      connection->errstr);

  if (!opt_.password.empty()) {
    helper::ReplyPtr reply;
    if (opt_.username.empty()) {
      reply = helper::ReplyPtr(
          reinterpret_cast<redisReply*>(redisCommand(
              connection.get(), "AUTH %s", opt_.password.c_str())));
    } else {
      reply = helper::ReplyPtr(
          reinterpret_cast<redisReply*>(redisCommand(
              connection.get(),
              "AUTH %s %s",
              opt_.username.c_str(),
              opt_.password.c_str())));
    }
    check_status("auth error", connection, reply);
  }

  if (opt_.db != 0) {
    auto reply = helper::ReplyPtr(
        reinterpret_cast<redisReply*>(
            redisCommand(connection.get(), "SELECT %d", opt_.db)));
    check_status("select db error", connection, reply);
  }

  return connection;
}

Redis::~Redis() {
  for (uint32_t i = 0; i < opt_.num_io_threads; ++i) {
    jobs_.emplace_back();
  }
  jobs_not_empty_.notify_all();
  for (auto& th : io_threads_) {
    th.join();
  }
}

static uint32_t CalculateChunkSizeByGlobalIDs(
    uint32_t chunk_size,
    uint32_t num_cols,
    uint32_t num_os) {
  static constexpr uint32_t low = 1;
  return std::max(
      chunk_size / std::max(num_cols, low) / std::max(num_os, low), low);
}

struct RedisFetchContext {
  std::atomic<uint32_t> num_complete_ids{0};
  uint32_t chunk_size;
  std::string table_name;
  std::vector<int64_t> global_ids;
  std::vector<int64_t> col_ids;
  uint32_t num_optimizer_states;
  void* on_complete_context;
  void (*on_global_id_fetched)(
      void* ctx,
      uint32_t gid_offset,
      uint32_t optimizer_state,
      void* data,
      uint32_t data_len);
  void (*on_all_fetched)(void* ctx);

  explicit RedisFetchContext(uint32_t chunk_size, IOFetchParameter param)
      : chunk_size(CalculateChunkSizeByGlobalIDs(
            chunk_size,
            param.num_cols,
            param.num_optimizer_states)),
        table_name(param.table_name),
        global_ids(param.global_ids, param.global_ids + param.num_global_ids),
        num_optimizer_states(param.num_optimizer_states),
        on_complete_context(param.on_complete_context),
        on_global_id_fetched(param.on_global_id_fetched),
        on_all_fetched(param.on_all_fetched) {
    if (param.num_cols == 0) {
      col_ids.emplace_back(-1);
    } else {
      col_ids =
          std::vector<int64_t>(param.col_ids, param.col_ids + param.num_cols);
    }
  }
};

void Redis::fetch(IOFetchParameter param) {
  auto* fetch_param = new RedisFetchContext(opt_.chunk_size, param);
  {
    std::lock_guard<std::mutex> guard(this->jobs_mutex_);
    for (uint32_t i = 0; i < param.num_global_ids;
         i += fetch_param->chunk_size) {
      jobs_.emplace_back(
          [i, fetch_param, this](helper::ContextPtr& connection) {
            do_fetch(i, fetch_param, connection);
          });
    }
  }
  jobs_not_empty_.notify_all();
}

void Redis::do_fetch(
    uint32_t gid_offset,
    void* fetch_param_void,
    helper::ContextPtr& connection) const {
  auto& fetch_param = *reinterpret_cast<RedisFetchContext*>(fetch_param_void);

  uint32_t end = std::min(
      gid_offset + fetch_param.chunk_size,
      static_cast<uint32_t>(fetch_param.global_ids.size()));

  auto loop = [&](auto&& callback) {
    for (uint32_t i = gid_offset; i < end; ++i) {
      int64_t gid = fetch_param.global_ids[i];
      for (uint32_t j = 0; j < fetch_param.col_ids.size(); ++j) {
        auto& col_id = fetch_param.col_ids[j];
        for (uint32_t os_id = 0; os_id < fetch_param.num_optimizer_states;
             ++os_id) {
          callback(i * fetch_param.col_ids.size() + j, gid, col_id, os_id);
        }
      }
    }
  };

  loop([&](uint32_t offset, int64_t gid, uint32_t col_id, uint32_t os_id) {
    redisAppendCommand(
        connection.get(),
        "GET %s_table_%s_gid_%d_cid_%d_osid_%d",
        opt_.prefix.c_str(),
        fetch_param.table_name.c_str(),
        gid,
        col_id,
        os_id);
  });

  void* reply;
  loop([&](uint32_t offset, int64_t gid, uint32_t col_id, uint32_t os_id) {
    int status = redisGetReply(connection.get(), &reply);
    TORCH_CHECK(
        status != REDIS_ERR,
        "get reply error: %s, from redis %s, %d",
        connection->errstr,
        opt_.host,
        opt_.port);
    auto reply_ptr = helper::ReplyPtr(reinterpret_cast<redisReply*>(reply));

    if (reply_ptr->type == REDIS_REPLY_NIL) {
      fetch_param.on_global_id_fetched(
          fetch_param.on_complete_context, offset, os_id, nullptr, 0);
    } else {
      fetch_param.on_global_id_fetched(
          fetch_param.on_complete_context,
          offset,
          os_id,
          reply_ptr->str,
          reply_ptr->len);
    }
  });

  uint32_t n = end - gid_offset;
  uint32_t target = fetch_param.global_ids.size();

  if (fetch_param.num_complete_ids.fetch_add(n) + n ==
      target) { // last fetch complete
    fetch_param.on_all_fetched(fetch_param.on_complete_context);
    delete &fetch_param;
  }
}

struct RedisPushContext {
  std::atomic<uint32_t> num_complete_ids{0};
  uint32_t chunk_size;
  std::string table_name;
  std::span<const int64_t> global_ids;
  std::vector<int64_t> col_ids;
  std::span<const uint32_t> os_ids;
  std::span<const uint64_t> offsets;
  const void* data;
  void* on_complete_context;
  void (*on_push_complete)(void*);

  RedisPushContext(uint32_t chunk_size, IOPushParameter param)
      : chunk_size(CalculateChunkSizeByGlobalIDs(
            chunk_size,
            param.num_cols,
            param.num_optimizer_states)),
        table_name(param.table_name),
        global_ids(param.global_ids, param.num_global_ids),
        os_ids(param.optimizer_state_ids, param.num_optimizer_states),
        offsets(param.offsets, param.num_offsets),
        data(param.data),
        on_complete_context(param.on_complete_context),
        on_push_complete(param.on_push_complete) {
    if (param.num_cols != 0) {
      col_ids =
          std::vector<int64_t>(param.col_ids, param.col_ids + param.num_cols);
    } else {
      col_ids.emplace_back(-1);
    }
  }
};

void Redis::push(IOPushParameter param) {
  auto* ctx = new RedisPushContext(opt_.chunk_size, param);
  {
    std::lock_guard<std::mutex> guard(this->jobs_mutex_);
    for (uint32_t i = 0; i < param.num_global_ids; i += ctx->chunk_size) {
      jobs_.emplace_back([i, ctx, this](helper::ContextPtr& connection) {
        do_push(i, ctx, connection);
      });
    }
  }
  jobs_not_empty_.notify_all();
}
void Redis::do_push(
    uint32_t gid_offset,
    void* push_ctx_ptr,
    helper::ContextPtr& connection) const {
  auto& push_ctx = *reinterpret_cast<RedisPushContext*>(push_ctx_ptr);

  uint32_t end = gid_offset + push_ctx.chunk_size;
  if (end > push_ctx.global_ids.size()) {
    end = push_ctx.global_ids.size();
  }

  auto loop = [&](auto&& callback) {
    for (uint32_t i = gid_offset; i < end; ++i) {
      int64_t gid = push_ctx.global_ids[i];
      for (uint32_t j = 0; j < push_ctx.col_ids.size(); ++j) {
        int64_t cid = push_ctx.col_ids[j];
        for (uint32_t k = 0; k < push_ctx.os_ids.size(); ++k) {
          uint32_t os_id = push_ctx.os_ids[k];

          uint32_t offset = k + j * push_ctx.os_ids.size() +
              i * push_ctx.col_ids.size() * push_ctx.os_ids.size();
          callback(offset, gid, cid, os_id);
        }
      }
    }
  };

  loop([&](uint32_t o, int64_t gid, int64_t cid, uint32_t os_id) {
    uint64_t beg = push_ctx.offsets[o];
    uint64_t end = push_ctx.offsets[o + 1];

    redisAppendCommand(
        connection.get(),
        "SET %s_table_%s_gid_%d_cid_%d_osid_%d %b",
        opt_.prefix.c_str(),
        push_ctx.table_name.c_str(),
        gid,
        cid,
        os_id,
        reinterpret_cast<const uint8_t*>(push_ctx.data) + beg,
        static_cast<size_t>(end - beg));
  });

  void* replay_ptr;
  loop([&](...) {
    int status = redisGetReply(connection.get(), &replay_ptr);
    TORCH_CHECK(
        status != REDIS_ERR,
        "get reply error: %s, from redis %s, %d",
        connection->errstr,
        opt_.host,
        opt_.port);
    helper::ReplyPtr reply(reinterpret_cast<redisReply*>(replay_ptr));
    check_status("reply should be ok", connection, reply);
  });

  uint32_t n = end - gid_offset;
  uint32_t target = push_ctx.global_ids.size();
  if (push_ctx.num_complete_ids.fetch_add(n) + n == target) {
    push_ctx.on_push_complete(push_ctx.on_complete_context);
    delete &push_ctx;
  }
}
void Redis::check_status(
    std::string_view label,
    helper::ContextPtr& connection,
    helper::ReplyPtr& reply) const {
  TORCH_CHECK(
      connection->err == 0,
      label,
      " connection error: (",
      connection->errstr,
      "), from redis://",
      opt_.host,
      ":",
      opt_.port);

  TORCH_CHECK(
      reply->type == REDIS_REPLY_STATUS,
      label,
      " reply should be status, but actual type is ",
      reply->type,
      ". from redis://",
      opt_.host,
      ":",
      opt_.port);

  auto status = std::string_view{reply->str, reply->len};
  TORCH_CHECK(
      status == "OK",
      label,
      " reply status should be OK, but actual is ",
      status,
      ". from redis://",
      opt_.host,
      ":",
      opt_.port);
}

extern "C" {

const char* IO_type = "redis";

void* IO_Initialize(const char* cfg) {
  auto opt = parse_option(cfg);
  return new Redis(opt);
}

void IO_Finalize(void* instance) {
  delete reinterpret_cast<Redis*>(instance);
}

void IO_Fetch(void* instance, IOFetchParameter param) {
  reinterpret_cast<Redis*>(instance)->fetch(param);
}

void IO_Push(void* instance, IOPushParameter param) {
  reinterpret_cast<Redis*>(instance)->push(param);
}
}

} // namespace torchrec::redis
