#include "gtest/gtest.h"
#include "tde/details/io.h"
#include "tde/details/notification.h"

namespace tde::details {
static int _r = [] {
  IORegistry::RegisterAllDefaultIOs();
  return 0;
}();

TEST(TDE, IO_redis) {
  constexpr static int64_t global_ids[] = {1, 3, 4};
  constexpr static uint32_t os_ids[] = {0};
  constexpr static float params[] = {1, 2, 3, 4, 5, 9, 8, 1};
  constexpr static uint64_t offsets[] = {
      0 * sizeof(float),
      2 * sizeof(float),
      4 * sizeof(float),
      6 * sizeof(float),
      8 * sizeof(float)};

  IO io("redis://127.0.0.1:6379/?prefix=io_redis_test");
  Notification notification;
  io.Push(
      "table",
      tcb::span<const int64_t>(
          global_ids, sizeof(global_ids) / sizeof(global_ids[0])),
      {},
      tcb::span<const uint32_t>(os_ids, sizeof(os_ids) / sizeof(os_ids[0])),
      tcb::span<const uint8_t>(
          reinterpret_cast<const uint8_t*>(params), sizeof(params)),
      tcb::span<const uint64_t>(offsets, sizeof(offsets) / sizeof(offsets[0])),
      [&notification] { notification.Done(); });
  notification.Wait();

  notification.Clear();
  io.Pull(
      "table",
      tcb::span<const int64_t>(
          global_ids, sizeof(global_ids) / sizeof(global_ids[0])),
      {},
      1,
      torch::kF32,
      [&](std::vector<torch::Tensor> val) {
        ASSERT_EQ(val.size(), 3);
        ASSERT_TRUE(
            val[0].allclose(
                torch::tensor(
                    {1, 2}, torch::TensorOptions().dtype(torch::kF32))));
        ASSERT_TRUE(
            val[1].allclose(
                torch::tensor(
                    {3, 4}, torch::TensorOptions().dtype(torch::kF32))));
        ASSERT_TRUE(
            val[2].allclose(
                torch::tensor(
                    {5, 9}, torch::TensorOptions().dtype(torch::kF32))));

        notification.Done();
      });
  notification.Wait();
}

} // namespace tde::details
