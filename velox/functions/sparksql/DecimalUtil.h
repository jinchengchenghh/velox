/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>
#include "velox/common/base/CheckedArithmetic.h"
#include "velox/common/base/Exceptions.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/Type.h"
#include "velox/type/UnscaledLongDecimal.h"
#include "velox/type/UnscaledShortDecimal.h"

namespace facebook::velox::sparksql {

class DecimalUtil {
 public:
  inline static int32_t maxBitsRequiredIncreaseAfterScaling(int32_t scale_by) {
    // We rely on the following formula:
    // bits_required(x * 10^y) <= bits_required(x) + floor(log2(10^y)) + 1
    // We precompute floor(log2(10^x)) + 1 for x = 0, 1, 2...75, 76

    static const int32_t floor_log2_plus_one[] = {
        0,   4,   7,   10,  14,  17,  20,  24,  27,  30,  34,  37,  40,
        44,  47,  50,  54,  57,  60,  64,  67,  70,  74,  77,  80,  84,
        87,  90,  94,  97,  100, 103, 107, 110, 113, 117, 120, 123, 127,
        130, 133, 137, 140, 143, 147, 150, 153, 157, 160, 163, 167, 170,
        173, 177, 180, 183, 187, 190, 193, 196, 200, 203, 206, 210, 213,
        216, 220, 223, 226, 230, 233, 236, 240, 243, 246, 250, 253};
    return floor_log2_plus_one[scale_by];
  }

  template <typename A>
  inline static int32_t maxBitsRequiredAfterScaling(
      const A& num,
      uint8_t aRescale) {
    auto value = num.unscaledValue();
    auto valueAbs = std::abs(value);
    int32_t num_occupied = 0;
    if constexpr (std::is_same_v<A, UnscaledShortDecimal>) {
      num_occupied = 64 - bits::countLeadingZeros(valueAbs);
    } else {
      num_occupied = 128 - num.countLeadingZeros();
    }

    return num_occupied + maxBitsRequiredIncreaseAfterScaling(aRescale);
  }

  template <typename R, typename A, typename B>
  inline static R divideWithRoundUp(
      R& r,
      const A& a,
      const B& b,
      uint8_t aRescale,
      uint8_t /*bRescale*/,
      bool* overflow) {
    if (b.unscaledValue() == 0) {
      *overflow = true;
      return R(-1);
    }
    int resultSign = 1;
    R unsignedDividendRescaled(a);
    int aSign = 1;
    int bSign = 1;
    if (a < 0) {
      resultSign = -1;
      unsignedDividendRescaled *= -1;
      aSign = -1;
    }
    R unsignedDivisor(b);
    if (b < 0) {
      resultSign *= -1;
      unsignedDivisor *= -1;
      bSign = -1;
    }
    auto bitsRequiredAfterScaling = maxBitsRequiredAfterScaling<A>(a, aRescale);
    if (bitsRequiredAfterScaling <= 127) {
      unsignedDividendRescaled = unsignedDividendRescaled.multiply(
          R(velox::DecimalUtil::kPowersOfTen[aRescale]), overflow);
      if (*overflow) {
        return R(-1);
      }
      R quotient = unsignedDividendRescaled / unsignedDivisor;
      R remainder = unsignedDividendRescaled % unsignedDivisor;
      if (remainder * 2 >= unsignedDivisor) {
        ++quotient;
      }
      r = quotient * resultSign;
      return remainder;
    } else {
      // Derives from Arrow BasicDecimal128 Divide
      if (aRescale > 38 && bitsRequiredAfterScaling > 255) {
        *overflow = true;
        return R(-1);
      }
      int256_t aLarge = a.unscaledValue();
      int256_t x_large_scaled_up =
          aLarge * velox::DecimalUtil::kPowersOfTen[aRescale];
      int256_t y_large = b.unscaledValue();
      int256_t resultLarge = x_large_scaled_up / y_large;
      int256_t remainder_large = x_large_scaled_up % y_large;
      // Since we are scaling up and then, scaling down, round-up the result (+1
      // for +ve, -1 for -ve), if the remainder is >= 2 * divisor.
      if (abs(2 * remainder_large) >= abs(y_large)) {
        // x +ve and y +ve, result is +ve =>   (1 ^ 1)  + 1 =  0 + 1 = +1
        // x +ve and y -ve, result is -ve =>  (-1 ^ 1)  + 1 = -2 + 1 = -1
        // x +ve and y -ve, result is -ve =>   (1 ^ -1) + 1 = -2 + 1 = -1
        // x -ve and y -ve, result is +ve =>  (-1 ^ -1) + 1 =  0 + 1 = +1
        resultLarge += (aSign ^ bSign) + 1;
      }

      auto result = R::convert(resultLarge, overflow);
      if (*overflow) {
        return R(-1);
      }
      r = result;
      auto remainder = R::convert(remainder_large, overflow);
      return remainder;
    }
  }
};
} // namespace facebook::velox::sparksql
