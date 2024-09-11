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

#include <iostream>

namespace facebook::velox::exec {
namespace {
bool canFullySupportByPrefixSort() {
  return true;
}

// Returns the metadata size
size_t metadataSize(
    const std::vector<TypePtr>& sortedTypes,
    const std::vector<column_index_t>& sortColumnIndices,
    std::vector<uint32_t>& prefixOffsets,
    uint32_t& prefixSize) {
  for (auto& type : sortedTypes) {
    auto size = prefixsort::PrefixSortEncoder::encodedSize(type->kind());
    VELOX_DCHECK(size.has_value());
    prefixSize += size.value();
    prefixOffsets.push_back(prefixSize);
  }
  // Prefix + buffer-idx + row-number.
  return prefixSize + sizeof(size_t) + sizeof(vector_size_t);
}

// @param prefixLength the prefix length to serialize, is a fixed size for this
// operator.
// @param vectorIdx write the value after prefix.
template <typename T>
FOLLY_ALWAYS_INLINE void encodeColumn(
    const DecodedVector& decoded,
    const prefixsort::PrefixSortEncoder& encoder,
    uint32_t prefixOffset,
    uint32_t prefixLength,
    int32_t padding,
    int32_t vectorIdx,
    vector_size_t numRows,
    char* const startAddress) {
  const auto metadataLength =
      prefixLength + sizeof(int32_t) + sizeof(vector_size_t);
  for (auto row = 0; row < numRows; row++) {
    char* prefix = startAddress + metadataLength * row + prefixOffset;
    if (decoded.isNullAt(row)) {
      encoder.encodeNull<T>(prefix);
    } else {
      encoder.encodeNoNulls<T>(decoded.valueAt<T>(row), prefix);
    }
    simd::memset(prefix + prefixLength - padding, 0, padding);
    PrefixSort::bitsSwapByWord((uint64_t*)prefix, prefixLength);
    memcpy(prefix + prefixLength, &vectorIdx, sizeof(int32_t));
    memcpy(
        prefix + prefixLength + sizeof(int32_t),
        &numRows,
        sizeof(vector_size_t));
  }
}

FOLLY_ALWAYS_INLINE void encodeColumnToPrefix(
    TypeKind typeKind,
    const DecodedVector& decoded,
    const prefixsort::PrefixSortEncoder& encoder,
    uint32_t prefixOffset,
    uint32_t prefixLength,
    int32_t padding,
    int32_t vectorIdx,
    vector_size_t numRows,
    char* const startAddress) {
  switch (typeKind) {
    case TypeKind::INTEGER: {
      encodeColumn<int32_t>(
          decoded,
          encoder,
          prefixOffset,
          prefixLength,
          padding,
          vectorIdx,
          numRows,
          startAddress);
      return;
    }
    case TypeKind::BIGINT: {
      encodeColumn<int64_t>(
          decoded,
          encoder,
          prefixOffset,
          prefixLength,
          padding,
          vectorIdx,
          numRows,
          startAddress);
      return;
    }
    case TypeKind::REAL: {
      encodeColumn<float>(
          decoded,
          encoder,
          prefixOffset,
          prefixLength,
          padding,
          vectorIdx,
          numRows,
          startAddress);
      return;
    }
    case TypeKind::DOUBLE: {
      encodeColumn<double>(
          decoded,
          encoder,
          prefixOffset,
          prefixLength,
          padding,
          vectorIdx,
          numRows,
          startAddress);
      return;
    }
    case TypeKind::TIMESTAMP: {
      encodeColumn<Timestamp>(
          decoded,
          encoder,
          prefixOffset,
          prefixLength,
          padding,
          vectorIdx,
          numRows,
          startAddress);
      return;
    }
    default:
      VELOX_UNSUPPORTED(
          "prefix-sort does not support type kind: {}",
          mapTypeKindToName(typeKind));
  }
}

std::vector<TypePtr> getSortedColumnTypes(
    const RowTypePtr& input,
    const std::vector<column_index_t>& sortColumnIndices,
    const std::vector<CompareFlags>& sortCompareFlags) {
  std::vector<TypePtr> sortedColumnTypes;
  sortedColumnTypes.reserve(sortColumnIndices.size());
  for (column_index_t i = 0; i < sortColumnIndices.size(); ++i) {
    sortedColumnTypes.emplace_back(input->childAt(sortColumnIndices.at(i)));
  }
  return sortedColumnTypes;
}

template <TypeKind Kind>
void setColumn(
    const DecodedVector& decoded,
    const std::vector<VectorPtr>& vectors,
    char* const startAddress,
    uint32_t entrySize,
    VectorPtr& output) {
  using T = typename velox::TypeTraits<Kind>::NativeType;
  const auto numRows = output->size();
  auto flat = output->asFlatVector<T>();
  for (auto i = 0; i < numRows; ++i) {
    char* prefixAddress = startAddress + i * entrySize;
    vector_size_t* bufferIdxAndRowNumber =
        reinterpret_cast<vector_size_t*>(prefixAddress);
    int32_t bufferIdx = bufferIdxAndRowNumber[0];
    vector_size_t rowNumber = bufferIdxAndRowNumber[1];
    if (decoded.isNullAt(i)) {
      output->setNull(i, true);
    } else {
      flat->set(i, decoded.valueAt<T>(rowNumber));
    }
  }
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
    : SortBuffer(
          input,
          sortColumnIndices,
          sortCompareFlags,
          pool,
          nonReclaimableSection,
          prefixSortConfig),
      layout_(PrefixSortLayout::makeSortLayout(
          getSortedColumnTypes(input, sortColumnIndices, sortCompareFlags),
          sortCompareFlags,
          prefixSortConfig.maxNormalizedKeySize,
          sizeof(size_t) + sizeof(vector_size_t))) // Vector-idx + row-number.
{
  std::unordered_set<column_index_t> sortedChannelSet;
  // Sorted key columns.
  for (column_index_t i = 0; i < sortColumnIndices.size(); ++i) {
    columnMap_.emplace_back(IdentityProjection(i, sortColumnIndices.at(i)));
  }
  // Non-sorted key columns.
  for (column_index_t i = 0, nonSortedIndex = sortCompareFlags_.size();
       i < input_->size();
       ++i) {
    if (sortedChannelSet.count(i) != 0) {
      continue;
    }
    columnMap_.emplace_back(nonSortedIndex++, i);
  }

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
  vectors_.emplace_back(input);
  numInputRows_ += input->size();
}

void SortBufferFast::noMoreInput() {
  VELOX_CHECK(!noMoreInput_);
  noMoreInput_ = true;

  // No data.
  if (numInputRows_ == 0) {
    return;
  }

  updateEstimatedOutputRowSize();
  const auto entrySize = layout_.entrySize;
  // 1. Allocate prefixes data.
  {
    const auto numPages =
        memory::AllocationTraits::numPages(numInputRows_ * entrySize);
    pool_->allocateContiguous(numPages, prefixAllocation_);
  }
  char* const prefixes = prefixAllocation_.data<char>();
  auto vectorSizeOffset = 0;
  for (auto i = 0; i < vectors_.size(); ++i) {
    const auto& vector = vectors_[i];
    serializeMetadata(vector, i, prefixes + vectorSizeOffset);
    vectorSizeOffset += vector->size() * entrySize;
  }
  // Sort the prefix in `buffers_`.
  const auto swapBuffer = AlignedBuffer::allocate<char>(entrySize, pool_);
  prefixsort::PrefixSortRunner sortRunner(
      entrySize, swapBuffer->asMutable<char>());
  const auto start = prefixes;
  const auto end = prefixes + numInputRows_ * entrySize;
  VELOX_DCHECK(!sortLayout_.hasNonNormalizedKey);
  sortRunner.quickSort(start, end, [&](char* a, char* b) {
    return PrefixSort::compareAllNormalizedKeys(
        a, b, layout_.normalizedBufferSize);
  });

  // Releases the unused memory reservation after procesing input.
  pool_->release();
}

RowVectorPtr SortBufferFast::getOutput(uint32_t maxOutputRows) {
  VELOX_CHECK(noMoreInput_);

  if (numOutputRows_ == numInputRows_) {
    return nullptr;
  }

  prepareOutput(maxOutputRows);
  // Reconstruct the RowVector by the record vector-idx and row-number
  char* const prefixes = prefixAllocation_.data<char>();
  char* const startAddress = prefixes + numOutputRows_ * layout_.entrySize;
  for (column_index_t col = 0; col < input_->size(); ++col) {
    VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        setColumn,
        input_->childAt(col)->kind(),
        decodedVectors_[col],
        vectors_,
        startAddress,
        layout_.entrySize,
        output_->childAt(col));
  }

  std::cout << "output data" << output_->toString(0, 20) << std::endl;
  return output_;
}

void SortBufferFast::updateEstimatedOutputRowSize() {
  uint64_t totalSize = 0;
  uint32_t estimateVectorNumber = 0;
  const auto divisor = std::max(1, static_cast<int32_t>(vectors_.size() / 10));
  for (auto i = 0; i < vectors_.size(); ++i) {
    if (i % divisor != 0) {
      continue;
    }
    totalSize += vectors_[i]->estimateFlatSize();
    estimateVectorNumber++;
  }
  estimatedOutputRowSize_ = totalSize / estimateVectorNumber;
}

void SortBufferFast::serializeMetadata(
    const VectorPtr& input,
    int32_t vectorIdx,
    char* const startMemoryAddress) {
  auto* inputRow = input->as<RowVector>();
  for (const auto& columnProjection : columnMap_) {
    const auto col = columnProjection.outputChannel;
    auto& decoded = decodedVectors_[col];
    decoded.decode(*inputRow->childAt(col));
    auto colType = inputRow->type()->childAt(col);
    const auto numRows = input->size();
    encodeColumnToPrefix(
        inputRow->type()->childAt(col)->kind(),
        decoded,
        encoders_[col],
        layout_.prefixOffsets[col],
        layout_.normalizedBufferSize,
        layout_.padding,
        vectorIdx,
        input->size(),
        startMemoryAddress);
  }
}
} // namespace facebook::velox::exec
