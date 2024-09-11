
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

#include "velox/exec/ContainerRowSerde.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/PrefixSort.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/SortBuffer.h"
#include "velox/exec/Spill.h"
#include "velox/row/UnsafeRowFast.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec {

/// A utility class to accumulate data inside and output the sorted result.
/// Spilling would be triggered if spilling is enabled and memory usage exceeds
/// limit.
class SortBufferFast : public SortBuffer {
 public:
  SortBufferFast(
      const RowTypePtr& input,
      const std::vector<column_index_t>& sortColumnIndices,
      const std::vector<CompareFlags>& sortCompareFlags,
      velox::memory::MemoryPool* pool,
      tsan_atomic<bool>* nonReclaimableSection,
      common::PrefixSortConfig prefixSortConfig,
      const common::SpillConfig* spillConfig = nullptr,
      folly::Synchronized<velox::common::SpillStats>* spillStats = nullptr);

  void addInput(const VectorPtr& input) override;

  /// Indicates no more input and triggers either of:
  ///  - In-memory sorting on rows stored in 'data_' if spilling is not enabled.
  ///  - Finish spilling and setup the sort merge reader for the un-spilling
  ///  processing for the output.
  void noMoreInput() override;

  /// Returns the sorted output rows in batch.
  RowVectorPtr getOutput(uint32_t maxOutputRows) override;

 private:
  // Ensures there is sufficient memory reserved to process 'input'.
  // void ensureInputFits(const VectorPtr& input);

  void init(const std::vector<column_index_t>& sortColumnIndices) override {};

  // Use the 10% input vectors to update estimate output row size.
  void updateEstimatedOutputRowSize() override;

  void serializeMetadata(
      const VectorPtr& input,
      int32_t vectorIdx,
      char* const startMemoryAddress);

  std::vector<prefixsort::PrefixSortEncoder> encoders_;
  std::vector<DecodedVector> decodedVectors_;

  std::vector<VectorPtr> vectors_;
  memory::ContiguousAllocation prefixAllocation_;

  const PrefixSortLayout layout_;
};

} // namespace facebook::velox::exec
