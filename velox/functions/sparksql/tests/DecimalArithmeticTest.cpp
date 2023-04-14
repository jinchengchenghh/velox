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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class DecimalArithmeticTest : public SparkFunctionBaseTest {
 public:
  DecimalArithmeticTest() {
    options_.parseDecimalAsDouble = false;
  }

 protected:
  template <TypeKind KIND>
  void testDecimalExpr(
      const VectorPtr& expected,
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    using EvalType = typename velox::TypeTraits<KIND>::NativeType;
    auto result =
        evaluate<SimpleVector<EvalType>>(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
    // testOpDictVectors<EvalType>(expression, expected, input);
  }

  template <typename T>
  void testOpDictVectors(
      const std::string& operation,
      const VectorPtr& expected,
      const std::vector<VectorPtr>& flatVector) {
    // Dictionary vectors as arguments.
    auto newSize = flatVector[0]->size() * 2;
    std::vector<VectorPtr> dictVectors;
    for (auto i = 0; i < flatVector.size(); ++i) {
      auto indices = makeIndices(newSize, [&](int row) { return row / 2; });
      dictVectors.push_back(
          VectorTestBase::wrapInDictionary(indices, newSize, flatVector[i]));
    }
    auto resultIndices = makeIndices(newSize, [&](int row) { return row / 2; });
    auto expectedResultDictionary =
        VectorTestBase::wrapInDictionary(resultIndices, newSize, expected);
    auto actual =
        evaluate<SimpleVector<T>>(operation, makeRowVector(dictVectors));
    assertEqualVectors(expectedResultDictionary, actual);
  }

  VectorPtr makeLongDecimalVector(
      const std::vector<std::string>& value,
      int8_t precision,
      int8_t scale) {
    std::vector<int128_t> int128s;
    for (auto& v : value) {
      bool nullOutput;
      int128s.emplace_back(convertStringToInt128(std::move(v), nullOutput));
      VELOX_CHECK(!nullOutput);
    }
    return makeLongDecimalFlatVector(int128s, DECIMAL(precision, scale));
  }

  int128_t convertStringToInt128(const std::string& value, bool& nullOutput) {
    // Handling integer target cases
    const char* v = value.c_str();
    nullOutput = true;
    bool negative = false;
    int128_t result = 0;
    int index = 0;
    int len = value.size();
    if (len == 0) {
      return -1;
    }
    // Setting negative flag
    if (v[0] == '-') {
      if (len == 1) {
        return -1;
      }
      negative = true;
      index = 1;
    }
    if (negative) {
      for (; index < len; index++) {
        if (!std::isdigit(v[index])) {
          return -1;
        }
        result = result * 10 - (v[index] - '0');
        // Overflow check
        if (result > 0) {
          return -1;
        }
      }
    } else {
      for (; index < len; index++) {
        if (!std::isdigit(v[index])) {
          return -1;
        }
        result = result * 10 + (v[index] - '0');
        // Overflow check
        if (result < 0) {
          return -1;
        }
      }
    }
    // Final result
    nullOutput = false;
    return result;
  }
}; // namespace

TEST_F(DecimalArithmeticTest, multiply) {
  // The result can be obtained by Spark unit test
  //       test("multiply") {
  //     val l1 = Literal.create(
  //       Decimal(BigDecimal(1), 17, 3),
  //       DecimalType(17, 3))
  //     val l2 = Literal.create(
  //       Decimal(BigDecimal(1), 17, 3),
  //       DecimalType(17, 3))
  //     checkEvaluation(Divide(l1, l2), null)
  //   }
  auto shortFlat = makeShortDecimalFlatVector({1000, 2000}, DECIMAL(17, 3));
  // Multiply short and short, returning long.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalFlatVector({1000000, 4000000}, DECIMAL(35, 6)),
      "multiply(c0, c1)",
      {shortFlat, shortFlat});
  // Multiply short and long, returning long.
  auto longFlat = makeLongDecimalFlatVector({1000, 2000}, DECIMAL(20, 3));
  auto expectedLongFlat =
      makeLongDecimalFlatVector({1000000, 4000000}, DECIMAL(38, 6));
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      expectedLongFlat, "multiply(c0, c1)", {shortFlat, longFlat});
  // Multiply long and short, returning long.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      expectedLongFlat, "multiply(c0, c1)", {longFlat, shortFlat});

  // Multiply long and long, returning long.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalFlatVector({1000000, 4000000}, DECIMAL(38, 6)),
      "multiply(c0, c1)",
      {longFlat, longFlat});

  auto leftFlat0 = makeLongDecimalFlatVector({0, 1, 0}, DECIMAL(20, 3));
  auto rightFlat0 = makeLongDecimalFlatVector({1, 0, 0}, DECIMAL(20, 2));
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalFlatVector({0, 0, 0}, DECIMAL(38, 5)),
      "multiply(c0, c1)",
      {leftFlat0, rightFlat0});

  // Multiply short and short, returning short.
  shortFlat = makeShortDecimalFlatVector({1000, 2000}, DECIMAL(6, 3));
  testDecimalExpr<TypeKind::SHORT_DECIMAL>(
      makeShortDecimalFlatVector({1000000, 4000000}, DECIMAL(13, 6)),
      "c0 * c1",
      {shortFlat, shortFlat});

  auto expectedConstantFlat =
      makeShortDecimalFlatVector({100000, 200000}, DECIMAL(10, 5));
  // Constant and Flat arguments.
  testDecimalExpr<TypeKind::SHORT_DECIMAL>(
      expectedConstantFlat, "1.00 * c0", {shortFlat});

  // Flat and Constant arguments.
  testDecimalExpr<TypeKind::SHORT_DECIMAL>(
      expectedConstantFlat, "c0 * 1.00", {shortFlat});

  // out_precision == 38, small input values, trimming of scale.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalFlatVector({61}, DECIMAL(38, 7)),
      "c0 * c1",
      {makeLongDecimalFlatVector({201}, DECIMAL(20, 5)),
       makeLongDecimalFlatVector({301}, DECIMAL(20, 5))});

  // out_precision == 38, large values, trimming of scale.
  bool nullOutput;
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalFlatVector(
          {convertStringToInt128("201" + std::string(31, '0'), nullOutput)},
          DECIMAL(38, 6)),
      "c0 * c1",
      {makeLongDecimalFlatVector({201}, DECIMAL(20, 5)),
       makeLongDecimalFlatVector(
           {convertStringToInt128(std::string(35, '9'), nullOutput)},
           DECIMAL(35, 5))});

  // out_precision == 38, very large values, trimming of scale (requires convert
  // to 256).
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalFlatVector(
          {convertStringToInt128(
              "9999999999999999999999999999999999890", nullOutput)},
          DECIMAL(38, 6)),
      "c0 * c1",
      {makeLongDecimalFlatVector(
           {convertStringToInt128(std::string(35, '9'), nullOutput)},
           DECIMAL(38, 20)),
       makeLongDecimalFlatVector(
           {convertStringToInt128(std::string(36, '9'), nullOutput)},
           DECIMAL(38, 20))});

  // out_precision == 38, very large values, trimming of scale (requires convert
  // to 256). should cause overflow.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeNullableLongDecimalFlatVector({std::nullopt}, DECIMAL(38, 6)),
      "c0 * c1",
      {makeLongDecimalFlatVector(
           {convertStringToInt128(std::string(35, '9'), nullOutput)},
           DECIMAL(38, 4)),
       makeLongDecimalFlatVector(
           {convertStringToInt128(std::string(36, '9'), nullOutput)},
           DECIMAL(38, 4))});

  // Big scale * big scale, mismatch with spark result
  // 0.0000060501000000000000000000000000000,
  // we cannot handle this case, it's intermediate result is beyond int256_t.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeNullableLongDecimalFlatVector({0}, DECIMAL(38, 37)),
      "c0 * c1",
      {makeLongDecimalFlatVector({201}, DECIMAL(38, 38)),
       makeLongDecimalFlatVector({301}, DECIMAL(38, 38))});

  // Long decimal limits
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeNullableLongDecimalFlatVector({std::nullopt}, DECIMAL(38, 0)),
      "c0 * cast(10.00 as decimal(2,0))",
      {makeLongDecimalFlatVector(
          {buildInt128(0x08FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)},
          DECIMAL(38, 0))});

  // Rescaling the final result overflows.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeNullableLongDecimalFlatVector({std::nullopt}, DECIMAL(38, 1)),
      "c0 * cast(1.00 as decimal(2,1))",
      {makeLongDecimalFlatVector(
          {buildInt128(0x08FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)},
          DECIMAL(38, 0))});
}

TEST_F(DecimalArithmeticTest, tmp) {
  auto shortFlat = makeShortDecimalFlatVector({1000, 2000}, DECIMAL(17, 3));
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeNullableLongDecimalFlatVector({std::nullopt}, DECIMAL(21, 6)),
      "c0 / 0.0",
      {shortFlat});
}

TEST_F(DecimalArithmeticTest, decimalDivTest) {
  auto shortFlat = makeShortDecimalFlatVector({1000, 2000}, DECIMAL(17, 3));
  // Divide short and short, returning long.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalVector(
          {"500000000000000000000", "2000000000000000000000"}, 38, 21),
      "divide(c0, c1)",
      {makeShortDecimalFlatVector({500, 4000}, DECIMAL(17, 3)), shortFlat});

  // Divide short and long, returning long.
  auto longFlat = makeLongDecimalFlatVector({500, 4000}, DECIMAL(20, 2));
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalFlatVector(
          {500000000000000000, 2000000000000000000}, DECIMAL(38, 17)),
      "divide(c0, c1)",
      {longFlat, shortFlat});

  // Divide long and short, returning long.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalVector(
          {"20" + std::string(20, '0'), "5" + std::string(20, '0')}, 38, 22),
      "divide(c0, c1)",
      {shortFlat, longFlat});

  // Divide long and long, returning long.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalVector(
          {"5" + std::string(18, '0'), "3" + std::string(18, '0')}, 38, 18),
      "divide(c0, c1)",
      {makeLongDecimalFlatVector({2500, 12000}, DECIMAL(20, 2)), longFlat});

  // Divide short and short, returning short.
  testDecimalExpr<TypeKind::SHORT_DECIMAL>(
      makeShortDecimalFlatVector({500000000, 300000000}, DECIMAL(13, 11)),
      "divide(c0, c1)",
      {makeShortDecimalFlatVector({2500, 12000}, DECIMAL(5, 5)),
       makeShortDecimalFlatVector({500, 4000}, DECIMAL(5, 2))});
  // This result can be obtained by Spark unit test
  //   test("divide decimal big") {
  //     val s = Seq(35, 6, 20, 3)
  //     var builder = new StringBuffer()
  //     (0 until 29).foreach(_ => builder = builder.append("9"))
  //     builder.append(".")
  //     (0 until 6).foreach(_ => builder = builder.append("9"))
  //     val str1 = builder.toString

  //     val l1 = Literal.create(
  //       Decimal(BigDecimal(str1), s.head, s(1)),
  //       DecimalType(s.head, s(1)))
  //     val l2 = Literal.create(
  //       Decimal(BigDecimal(0.201), s(2), s(3)),
  //       DecimalType(s(2), s(3)))
  //     checkEvaluation(Divide(l1, l2), null)
  //   }
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalVector({"497512437810945273631840796019900493"}, 38, 6),
      "c0 / c1",
      {makeLongDecimalVector({std::string(35, '9')}, 35, 6),
       makeLongDecimalFlatVector({201}, DECIMAL(20, 3))});

  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalVector(
          {"1000" + std::string(17, '0'), "500" + std::string(17, '0')},
          24,
          20),
      "1.00 / c0",
      {shortFlat});

  // Flat and Constant arguments.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeLongDecimalVector(
          {"500" + std::string(4, '0'), "1000" + std::string(4, '0')}, 23, 7),
      "c0 / 2.00",
      {shortFlat});

  // Divide and round-up.
  // The result can be obtained by Spark unit test
  //     test("divide test") {
  //     spark.sql("create table decimals_test(a decimal(2,1)) using
  //     parquet;") spark.sql("insert into decimals_test values(6)") val df =
  //     spark.sql("select a / -6.0 from decimals_test") df.collect()
  //     df.printSchema()
  //     df.show(truncate = false)
  //     print(df.queryExecution.executedPlan)
  //     spark.sql("drop table decimals_test;")
  //   }
  testDecimalExpr<TypeKind::SHORT_DECIMAL>(
      {makeShortDecimalFlatVector(
          {566667, -83333, -1083333, -1500000, -33333, 816667}, DECIMAL(8, 6))},
      "c0 / -6.0",
      {makeShortDecimalFlatVector({-34, 5, 65, 90, 2, -49}, DECIMAL(2, 1))});
  // Divide by zero.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeNullableLongDecimalFlatVector(
          {std::nullopt, std::nullopt}, DECIMAL(21, 6)),
      "c0 / 0.0",
      {shortFlat});

  // Long decimal limits.
  testDecimalExpr<TypeKind::LONG_DECIMAL>(
      makeNullableLongDecimalFlatVector({std::nullopt}, DECIMAL(38, 6)),
      "c0 / 0.01",
      {makeLongDecimalFlatVector(
          {UnscaledLongDecimal::max().unscaledValue()}, DECIMAL(38, 0))});
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
