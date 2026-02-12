/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <exception>

#define TORCHREC_INTERNAL_ASSERT_WITH_MESSAGE(condition, message)          \
  if (!(condition)) {                                                      \
    throw std::runtime_error(                                              \
        "Internal Assertion failed: (" + std::string(#condition) + "), " + \
        "function " + __FUNCTION__ + ", file " + __FILE__ + ", line " +    \
        std::to_string(__LINE__) + ".\n" +                                 \
        "Please report bug to TorchRec.\n" + message + "\n");              \
  }

#define TORCHREC_INTERNAL_ASSERT_NO_MESSAGE(condition) \
  TORCHREC_INTERNAL_ASSERT_WITH_MESSAGE(#condition, "")

#define TORCHREC_INTERNAL_ASSERT_(x, condition, message, FUNC, ...) FUNC

#define TORCHREC_INTERNAL_ASSERT(...)                     \
  TORCHREC_INTERNAL_ASSERT_(                              \
      ,                                                   \
      ##__VA_ARGS__,                                      \
      TORCHREC_INTERNAL_ASSERT_WITH_MESSAGE(__VA_ARGS__), \
      TORCHREC_INTERNAL_ASSERT_NO_MESSAGE(__VA_ARGS__));

#define TORCHREC_CHECK_WITH_MESSAGE(condition, message)                     \
  if (!(condition)) {                                                       \
    throw std::runtime_error(                                               \
        "Check failed: (" + std::string(#condition) + "), " + "function " + \
        __FUNCTION__ + ", file " + __FILE__ + ", line " +                   \
        std::to_string(__LINE__) + ".\n" + message + "\n");                 \
  }

#define TORCHREC_CHECK_NO_MESSAGE(condition) \
  TORCHREC_CHECK_WITH_MESSAGE(condition, "")

#define TORCHREC_CHECK_(x, condition, message, FUNC, ...) FUNC

#define TORCHREC_CHECK(...)                     \
  TORCHREC_CHECK_(                              \
      ,                                         \
      ##__VA_ARGS__,                            \
      TORCHREC_CHECK_WITH_MESSAGE(__VA_ARGS__), \
      TORCHREC_CHECK_NO_MESSAGE(__VA_ARGS__));
