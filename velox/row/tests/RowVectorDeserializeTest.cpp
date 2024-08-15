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

#include <gtest/gtest.h>

#include <folly/Random.h>
#include <folly/init/Init.h>

#include "velox/row/UnsafeRowDeserializers.h"
#include "velox/row/UnsafeRowFast.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::row {
namespace {

using namespace facebook::velox::test;

class RowVectorDeserializeTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  RowVectorDeserializeTest() {
    clearBuffers();
  }

 protected:
  void clearBuffers() {
    std::memset(buffer_, 0, kBufferSize);
    offsets_.clear();
  }

  void doTest(const RowTypePtr& rowType) {
    const vector_size_t numRows = 100;
    VectorFuzzer::Options opts;
    opts.vectorSize = numRows;
    opts.nullRatio = 0.1;
    opts.dictionaryHasNulls = false;
    opts.stringVariableLength = true;
    opts.stringLength = 20;
    opts.containerVariableLength = true;
    opts.complexElementsMaxSize = 10'000;

    // Spark uses microseconds to store timestamp
    opts.timestampPrecision =
        VectorFuzzer::Options::TimestampPrecision::kMicroSeconds,
    opts.containerLength = 10;

    VectorFuzzer fuzzer(opts, pool_.get());

    const auto iterations = 200;
    for (size_t i = 0; i < iterations; ++i) {
      clearBuffers();

      auto seed = folly::Random::rand32();
      fuzzer.reSeed(seed);
      const auto& inputVector = fuzzer.fuzzInputRow(rowType);

      UnsafeRowFast fast(inputVector);
      size_t offset = 0;
      offsets_.resize(numRows);
      for (auto i = 0; i < numRows; ++i) {
        auto rowSize = fast.serialize(i, (char*)buffer_ + offset);
        offsets_[i] = offset;
        offset += rowSize;
      }
      VELOX_CHECK_LE(offset, kBufferSize);

      // Deserialize previous bytes back to row vector
      VectorPtr outputVector = UnsafeRowDeserializer::deserialize(
          reinterpret_cast<const uint8_t*>(buffer_),
          rowType,
          offsets_,
          pool_.get());

      assertEqualVectors(inputVector, outputVector);
    }
  }

  static constexpr uint64_t kBufferSize = 7000 << 10; // 7Mb

  char buffer_[kBufferSize];
  std::vector<size_t> offsets_;

  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
};

TEST_F(RowVectorDeserializeTest, fast) {
  auto rowType = ROW({
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      VARCHAR(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      VARCHAR(),
      VARBINARY(),
      UNKNOWN(),
      DECIMAL(20, 2),
      DECIMAL(12, 4),
      TIMESTAMP(),
      DATE(),
  });

  doTest(rowType);
}

} // namespace
} // namespace facebook::velox::row
