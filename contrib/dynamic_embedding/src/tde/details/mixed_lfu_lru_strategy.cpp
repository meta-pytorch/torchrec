#include "mixed_lfu_lru_strategy.h"
#include <algorithm>
#include "c10/macros/Macros.h"

namespace tde::details {
MixedLFULRUStrategy::MixedLFULRUStrategy(uint16_t min_used_freq_power)
    : min_lfu_power_(min_used_freq_power), time_(new std::atomic<uint32_t>()) {}

void MixedLFULRUStrategy::UpdateTime(uint32_t time) {
  time_->store(time);
}

MixedLFULRUStrategy::lxu_record_t MixedLFULRUStrategy::Update(
    int64_t global_id,
    int64_t cache_id,
    std::optional<lxu_record_t> val) {
  Record r{};
  r.time_ = time_->load();

  if (!val.has_value()) [[unlikely]] {
    r.freq_power_ = min_lfu_power_;
  } else {
    auto freq_power = reinterpret_cast<Record*>(&val.value())->freq_power_;
    bool should_carry = generator_.IsNextNBitsAllZero(freq_power);
    if (should_carry) {
      ++freq_power;
    }
    r.freq_power_ = freq_power;
  }
  return *reinterpret_cast<lxu_record_t*>(&r);
}

} // namespace tde::details
