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

#include "velox/type/HugeInt.h"
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"

using namespace facebook;
using namespace facebook::velox;

namespace {
void testBasic(int128_t hugeInt, uint64_t upper, uint64_t lower) {
  EXPECT_EQ(hugeInt, HugeInt::build(upper, lower));
  EXPECT_EQ(upper, HugeInt::upper(hugeInt));
  EXPECT_EQ(lower, HugeInt::lower(hugeInt));
}

void testParse(int128_t hugeInt, const std::string& hugeString) {
  EXPECT_EQ(hugeInt, HugeInt::parse(hugeString));
  EXPECT_EQ(hugeString, std::to_string(hugeInt));
}

void testCombine(int64_t hi, int64_t lo, const std::string& hugeString) {
  EXPECT_EQ(std::to_string(HugeInt::combine(hi, lo)), hugeString);
}
} // namespace

TEST(HugeIntTest, basic) {
  testBasic(0xDEADBEEE, 0x0, 0xDEADBEEE);

  // 0xF{16}F{16} = -1
  auto uint64Max = static_cast<int128_t>(std::numeric_limits<uint64_t>::max());
  int128_t hugeInt = -1;
  testBasic(hugeInt, uint64Max, uint64Max);

  hugeInt = std::numeric_limits<int128_t>::max() - 0x12345;
  uint64_t upper = 0x7FFFFFFFFFFFFFFF;
  uint64_t lower = 0xFFFFFFFFFFFEDCBA;
  testBasic(hugeInt, upper, lower);

  hugeInt = std::numeric_limits<int128_t>::min() + 0xDEADBEEFCAFECAFE;
  upper = 0x8000000000000000;
  lower = 0xDEADBEEFCAFECAFE;
  testBasic(hugeInt, upper, lower);

  // uint64Max * 0xDEADBEEF + 0xBADFEED = 0x0{8}DEADBEEEF{8}2D003FFE
  hugeInt = uint64Max * 0xDEADBEEF + 0xBADFEED;
  testBasic(hugeInt, 0xDEADBEEE, 0xFFFFFFFF2D003FFE);
}

TEST(HugeIntTest, parse) {
  testParse(0, "0");
  testParse(13579, "13579");
  testParse(-13579, "-13579");

  const std::string kInt128MaxString =
      "170141183460469231731687303715884105727";
  auto hugeInt = std::numeric_limits<int128_t>::max();
  testParse(hugeInt, kInt128MaxString);

  const std::string kInt128MinString =
      "-170141183460469231731687303715884105728";
  hugeInt = std::numeric_limits<int128_t>::min();
  testParse(hugeInt, kInt128MinString);

  // uint64Max * 0xDEADBEEF + 0xBADFEED = 0x0{8}DEADBEEEF{8}2D003FFE =
  // 68915718005535514949759025150
  hugeInt = HugeInt::build(0xDEADBEEE, 0xFFFFFFFF2D003FFE);
  testParse(hugeInt, "68915718005535514949759025150");

  // 0x80{15} + 0xDEADBEEFBADFEEDDEADBEEFBADFEED =
  // 0x80DEADBEEFBADFEEDDEADBEEFBADFEED =
  // -168984969573469355505154650711096688915
  hugeInt = HugeInt::build(0x80DEADBEEFBADFEE, 0xDDEADBEEFBADFEED);
  testParse(hugeInt, "-168984969573469355505154650711096688915");

  VELOX_ASSERT_THROW(
      HugeInt::parse("1A"), "Invalid character A in the string.");
  VELOX_ASSERT_THROW(
      HugeInt::parse(""), "Empty string cannot be converted to int128_t");
  VELOX_ASSERT_THROW(
      testParse(hugeInt, "-170141183460469231731687303715884105729"),
      "out of range of int128_t");
  VELOX_ASSERT_THROW(
      testParse(hugeInt, "170141183460469231731687303715884105728"),
      "out of range of int128_t");
  VELOX_ASSERT_THROW(
      testParse(hugeInt, "170141183460469231731687303715884105730"),
      "out of range of int128_t");
}

TEST(HugeIntTest, combine) {
  testCombine(0, 0, "0");
  testCombine(0, 13579, "13579");
  testCombine(0, -13579, "-13579");
  testCombine(1, 13579, "1" + std::string(13, '0') + "13579");
  testCombine(-1, -13579, "-1" + std::string(13, '0') + "13579");
  testCombine(13579, 0, "13579" + std::string(18, '0'));
  testCombine(-13579, 0, "-13579" + std::string(18, '0'));

  testCombine(
      999'999'999'999'999'999L, 999'999'999'999'999'999L, std::string(36, '9'));

  testCombine(
      -999'999'999'999'999'999,
      -999'999'999'999'999'999,
      "-" + std::string(36, '9'));

  testCombine(INT64_MAX, INT64_MAX, "9223372036854775816223372036854775807");
  testCombine(INT64_MIN, INT64_MIN, "-9223372036854775816223372036854775807");

  // uint64Max * 0xDEADBEEF + 0xBADFEED = 0x0{8}DEADBEEEF{8}2D003FFE =
  // 68915718005535514949759025150
  testCombine(
      68'915'718'005, 535'514'949'759'025'150, "68915718005535514949759025150");

  VELOX_ASSERT_THROW(
      testCombine(12, -3, "xxx"), "High 12 and low -3 should have same symbol");
}
