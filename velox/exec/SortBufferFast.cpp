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

#include "SortBufferFast.h"
#include "velox/exec/MemoryReclaimer.h"

namespace facebook::velox::exec {
namespace {
bool canFullySupportByPrefixSort() {
  return true;
}

// Returns the serialized data size.
size_t serializedSize(const VectorPtr& vector, row::UnsafeRowFast* fast) {
  size_t totalSize = 0;
  if (auto fixedRowSize = velox::row::UnsafeRowFast::fixedRowSize(
          velox::asRowType(vector->type()))) {
    totalSize = vector->size() * fixedRowSize.value();
  } else {
    for (auto i = 0; i < vector->size(); ++i) {
      totalSize += fast->rowSize(i);
    }
  }
  return totalSize;
}

// Returns the metadata size
size_t metadataSize(
    const std::vector<TypePtr>& sortedTypes,
    const std::vector<column_index_t>& sortColumnIndices,
    std::vector<uint32_t>& prefixOffsets,
    uint32_t& prefixSize) {
  size_t prefixSize = 0;
  for (auto& type : sortedTypes) {
    auto size = prefixsort::PrefixSortEncoder::encodedSize(type->kind());
    VELOX_DCHECK(size.has_value());
    prefixSize += size.value();
    prefixOffsets.push_back(prefixSize);
  }
  // Prefix + buffer-idx + row-number.
  return prefixSize + sizeof(size_t) + sizeof(vector_size_t);
}

template <typename T>
FOLLY_ALWAYS_INLINE void encodeRowColumn(
    const DecodedVector& decoded,
    const prefixsort::PrefixSortEncoder& encoder,
    char* const row,
    char* const prefix) {
  if (decoded.isNullAt(row)) {
    encoder.encodeNull(prefix);
  } else {
    encoder.encode<T>(decoded.valueAt<T>(row), prefix);
  }
}

// @param prefixLength the prefix length to serialize, is a fixed size for this
// operator.
// @param bufferIdx write the value after prefix.
template <typename T>
FOLLY_ALWAYS_INLINE void encodeColumn(
    const DecodedVector& decoded,
    const prefixsort::PrefixSortEncoder& encoder,
    const std::vector<size_t> rowOffsets,
    uint32_t prefixOffset,
    uint32_t prefixLength,
    int32_t bufferIdx,
    vector_size_t numRows,
    const char* startAddress) {
  for (auto row = 0; row < numRows; row++) {
    char* prefix = startAddress + rowOffsets[row] + prefixOffset;
    if (decoded.isNullAt(row)) {
      encoder.encodeNull<T>(prefix);
    } else {
      encoder.encodeNoNulls<T>(prefix);
    }
    memcpy(prefix + prefixLength, &bufferIdx, sizeof(int32_t));
    memcpy(
        prefix + prefixLength + sizeof(int32_t),
        &bufferIdx,
        sizeof(vector_size_t));
  }
}

FOLLY_ALWAYS_INLINE void encodeColumnToPrefix(
    TypeKind typeKind,
    const DecodedVector& decoded,
    const prefixsort::PrefixSortEncoder& encoder,
    const std::vector<size_t> rowOffsets,
    uint32_t prefixOffset,
    uint32_t prefixLength,
    int32_t bufferIdx,
    vector_size_t numRows,
    const char* startAddress) {
  switch (typeKind) {
    case TypeKind::INTEGER: {
      encodeColumn<int32_t>(
          decoded,
          encoder,
          rowOffsets,
          prefixOffset,
          prefixLength,
          bufferIdx,
          numRows,
          startAddress);
      return;
    }
    case TypeKind::BIGINT: {
      encodeColumn<int64_t>(
          decoded,
          encoder,
          rowOffsets,
          prefixOffset,
          prefixLength,
          bufferIdx,
          numRows,
          startAddress);
      return;
    }
    case TypeKind::REAL: {
      encodeColumn<float>(
          decoded,
          encoder,
          rowOffsets,
          prefixOffset,
          prefixLength,
          bufferIdx,
          numRows,
          startAddress);
      return;
    }
    case TypeKind::DOUBLE: {
      encodeColumn<double>(
          decoded,
          encoder,
          rowOffsets,
          prefixOffset,
          prefixLength,
          bufferIdx,
          numRows,
          startAddress);
      return;
    }
    case TypeKind::TIMESTAMP: {
      encodeColumn<Timestamp>(
          decoded,
          encoder,
          rowOffsets,
          prefixOffset,
          prefixLength,
          bufferIdx,
          numRows,
          startAddress);
      return;
    }
    default:
      VELOX_UNSUPPORTED(
          "prefix-sort does not support type kind: {}",
          mapTypeKindToName(typeKind));
  }

} // namespace
SortBufferFast::SortBufferFast(
    const RowTypePtr& input,
    const std::vector<column_index_t>& sortColumnIndices,
    const std::vector<CompareFlags>& sortCompareFlags,
    velox::memory::MemoryPool* pool,
    tsan_atomic<bool>* nonReclaimableSection,
    common::PrefixSortConfig prefixSortConfig,
    const common::SpillConfig* spillConfig,
    folly::Synchronized<velox::common::SpillStats>* spillStats)
    : input_(input),
      sortCompareFlags_(sortCompareFlags),
      pool_(pool),
      nonReclaimableSection_(nonReclaimableSection),
      prefixSortConfig_(prefixSortConfig),
      fast_(std::make_unique<row::UnsafeRowFast>()),
      sortColumnIndices_(sortColumnIndices),
      SortBuffer(
          input,
          sortColumnIndices,
          sortCompareFlags,
          pool,
          nonReclaimableSection,
          prefixSortConfig) {
  VELOX_CHECK_GE(input_->size(), sortCompareFlags_.size());
  VELOX_CHECK_GT(sortCompareFlags_.size(), 0);
  VELOX_CHECK_EQ(sortColumnIndices.size(), sortCompareFlags_.size());
  VELOX_CHECK_NOT_NULL(nonReclaimableSection_);

  std::vector<TypePtr> sortedColumnTypes;
  std::vector<TypePtr> nonSortedColumnTypes;
  std::vector<std::string> sortedSpillColumnNames;
  std::vector<TypePtr> sortedSpillColumnTypes;
  sortedColumnTypes.reserve(sortColumnIndices.size());
  nonSortedColumnTypes.reserve(input->size() - sortColumnIndices.size());
  sortedSpillColumnNames.reserve(input->size());
  sortedSpillColumnTypes.reserve(input->size());
  std::unordered_set<column_index_t> sortedChannelSet;
  // Sorted key columns.
  for (column_index_t i = 0; i < sortColumnIndices.size(); ++i) {
    columnMap_.emplace_back(IdentityProjection(i, sortColumnIndices.at(i)));
    sortedColumnTypes.emplace_back(input_->childAt(sortColumnIndices.at(i)));
    sortedSpillColumnTypes.emplace_back(
        input_->childAt(sortColumnIndices.at(i)));
    sortedSpillColumnNames.emplace_back(input->nameOf(sortColumnIndices.at(i)));
    sortedChannelSet.emplace(sortColumnIndices.at(i));
  }
  // Non-sorted key columns.
  for (column_index_t i = 0, nonSortedIndex = sortCompareFlags_.size();
       i < input_->size();
       ++i) {
    if (sortedChannelSet.count(i) != 0) {
      continue;
    }
    columnMap_.emplace_back(nonSortedIndex++, i);
    nonSortedColumnTypes.emplace_back(input_->childAt(i));
    sortedSpillColumnTypes.emplace_back(input_->childAt(i));
    sortedSpillColumnNames.emplace_back(input->nameOf(i));
  }
  prefixOffsets_.reserve(sortColumnIndices.size());
  metadataSize_ = metadataSize(
      sortedColumnTypes, sortColumnIndices, prefixOffsets_, prefixLength_);
  {
    encoders_.reserve(sortCompareFlags.size());
    for (const auto& flag : sortCompareFlags) {
      encoders_.push_back({flag.ascending, flag.nullsFirst});
    }
  }
  decodedVectors_.resize(sortColumnIndices.size());
}

void SortBufferFast::addInput(const VectorPtr& input) {
  VELOX_CHECK(!noMoreInput_);
  auto totalSize =
      metadataSize_ * input->size() + serializedSize(input, fast_.get());
  BufferPtr buffer = velox::AlignedBuffer::allocate<uint8_t>(totalSize, pool_);
  buffers_.emplace_back(buffer);
  auto rawBuffer = buffer->asMutable<char>();
  size_t offset = 0;
  std::vector<size_t> offsets;
  offsets.resize(input->size());
  for (auto i = 0; i < input->size(); ++i) {
    offsets.emplace_back(offset);
    // Reserve the space for prefix and other metadata.
    offset += metadataSize_;
    auto rowSize = fast_->serialize(i, rawBuffer + offset);
    offset += rowSize;
  }
  offsets_.emplace_back(offsets);
  serializeMetadata(input, offsets, rawBuffer);
}

// Some one will optimize to serialize by column, then we don't need to
// serilialize each data switch all the types.
// TODO, change support type to one or serialize by column.
void SortBufferFast::serializeMetadata(
    const VectorPtr& input,
    const std::vector<size_t> offsets,
    const char* startMemoryAddress) {
  auto* inputRow = input->as<RowVector>();
  for (const auto& columnProjection : columnMap_) {
    auto col = columnProjection.outputChannel;
    auto& decoded = decodedVectors_[col];
    decoded.decode(*inputRow->childAt(col));
    auto colType = inputRow->type()->childAt(col);
    const auto numRows = input->size();
    encodeColumnToPrefix(
        inputRow->type()->childAt(col)->kind(),
        decoded,
        encoders_[col],
        offsets,
        prefixOffsets_[col],
        prefixLength_,
        buffers_.size() - 1,
        input->size(),
        startMemoryAddress);
  }
}
} // namespace facebook::velox::exec
