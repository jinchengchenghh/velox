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
#include "velox/row/UnsafeRowDeserializers.h"
// #include "velox/common/base/BitUtil.h"

namespace facebook::velox::row {
namespace {

inline int64_t getFieldOffset(int64_t nullBitsetWidthInBytes, int32_t index) {
  return nullBitsetWidthInBytes + UnsafeRow::kFieldWidthBytes * index;
}

inline bool isNullAt(const uint8_t* memoryAddress, int32_t index) {
  return bits::isBitSet(memoryAddress, index);
}

int32_t getTotalStringSize(
    int32_t columnIdx,
    int32_t numRows,
    int64_t fieldOffset,
    const std::vector<size_t>& offsets,
    const uint8_t* memoryAddress) {
  size_t size = 0;
  for (auto pos = 0; pos < numRows; pos++) {
    if (isNullAt(memoryAddress + offsets[pos], columnIdx)) {
      continue;
    }

    int64_t offsetAndSize =
        *(int64_t*)(memoryAddress + offsets[pos] + fieldOffset);
    int32_t length = static_cast<int32_t>(offsetAndSize);
    if (!StringView::isInline(length)) {
      size += length;
    }
  }
  return size;
}

template <TypeKind Kind>
VectorPtr createFlatVectorFast(
    const TypePtr& type,
    int32_t columnIdx,
    int32_t numRows,
    int64_t fieldOffset,
    const std::vector<size_t>& offsets,
    const uint8_t* memoryAddress,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<Kind>::NativeType;
  constexpr uint32_t typeWidth = sizeof(T);
  auto column = BaseVector::create<FlatVector<T>>(type, numRows, pool);
  auto rawValues = column->template mutableRawValues<uint8_t>();
  auto shift = __builtin_ctz(typeWidth);
  for (auto pos = 0; pos < numRows; pos++) {
    if (!isNullAt(memoryAddress + offsets[pos], columnIdx)) {
      const uint8_t* srcptr = (memoryAddress + offsets[pos] + fieldOffset);
      uint8_t* destptr = rawValues + (pos << shift);
      memcpy(destptr, srcptr, typeWidth);
    } else {
      column->setNull(pos, true);
    }
  }
  return column;
}

template <>
VectorPtr createFlatVectorFast<TypeKind::HUGEINT>(
    const TypePtr& type,
    int32_t columnIdx,
    int32_t numRows,
    int64_t fieldOffset,
    const std::vector<size_t>& offsets,
    const uint8_t* memoryAddress,
    memory::MemoryPool* pool) {
  auto column = BaseVector::create<FlatVector<int128_t>>(type, numRows, pool);
  auto rawValues = column->mutableRawValues<uint8_t>();
  constexpr uint32_t typeWidth = sizeof(int128_t);
  auto shift = __builtin_ctz(typeWidth);
  for (auto pos = 0; pos < numRows; pos++) {
    if (!isNullAt(memoryAddress + offsets[pos], columnIdx)) {
      uint8_t* destptr = rawValues + (pos << shift);
      int64_t offsetAndSize =
          *(int64_t*)(memoryAddress + offsets[pos] + fieldOffset);
      int32_t length = static_cast<int32_t>(offsetAndSize);
      int32_t wordoffset = static_cast<int32_t>(offsetAndSize >> 32);
      uint8_t bytesValue[length];
      memcpy(bytesValue, memoryAddress + offsets[pos] + wordoffset, length);
      uint8_t bytesValue2[16]{};
      for (int k = length - 1; k >= 0; k--) {
        bytesValue2[length - 1 - k] = bytesValue[k];
      }
      if (int8_t(bytesValue[0]) < 0) {
        memset(bytesValue2 + length, 255, 16 - length);
      }
      memcpy(destptr, bytesValue2, typeWidth);
    } else {
      column->setNull(pos, true);
    }
  }
  return column;
}

template <>
VectorPtr createFlatVectorFast<TypeKind::BOOLEAN>(
    const TypePtr& type,
    int32_t columnIdx,
    int32_t numRows,
    int64_t fieldOffset,
    const std::vector<size_t>& offsets,
    const uint8_t* memoryAddress,
    memory::MemoryPool* pool) {
  auto column = BaseVector::create<FlatVector<bool>>(type, numRows, pool);
  auto rawValues = column->mutableRawValues<uint64_t>();
  for (auto pos = 0; pos < numRows; pos++) {
    if (!isNullAt(memoryAddress + offsets[pos], columnIdx)) {
      bool value = *(bool*)(memoryAddress + offsets[pos] + fieldOffset);
      bits::setBit(rawValues, pos, value);
    } else {
      column->setNull(pos, true);
    }
  }
  return column;
}

template <>
VectorPtr createFlatVectorFast<TypeKind::TIMESTAMP>(
    const TypePtr& type,
    int32_t columnIdx,
    int32_t numRows,
    int64_t fieldOffset,
    const std::vector<size_t>& offsets,
    const uint8_t* memoryAddress,
    memory::MemoryPool* pool) {
  auto column = BaseVector::create<FlatVector<Timestamp>>(type, numRows, pool);
  for (auto pos = 0; pos < numRows; pos++) {
    if (!isNullAt(memoryAddress + offsets[pos], columnIdx)) {
      int64_t value = *(int64_t*)(memoryAddress + offsets[pos] + fieldOffset);
      column->set(pos, Timestamp::fromMicros(value));
    } else {
      column->setNull(pos, true);
    }
  }
  return column;
}

template <>
VectorPtr createFlatVectorFast<TypeKind::VARCHAR>(
    const TypePtr& type,
    int32_t columnIdx,
    int32_t numRows,
    int64_t fieldOffset,
    const std::vector<size_t>& offsets,
    const uint8_t* memoryAddress,
    memory::MemoryPool* pool) {
  auto column = BaseVector::create<FlatVector<StringView>>(type, numRows, pool);
  auto size = getTotalStringSize(
      columnIdx, numRows, fieldOffset, offsets, memoryAddress);
  char* rawBuffer = column->getRawStringBufferWithSpace(size, true);
  for (auto pos = 0; pos < numRows; pos++) {
    if (!isNullAt(memoryAddress + offsets[pos], columnIdx)) {
      int64_t offsetAndSize =
          *(int64_t*)(memoryAddress + offsets[pos] + fieldOffset);
      int32_t length = static_cast<int32_t>(offsetAndSize);
      int32_t wordoffset = static_cast<int32_t>(offsetAndSize >> 32);
      auto valueSrcPtr = memoryAddress + offsets[pos] + wordoffset;
      if (StringView::isInline(length)) {
        column->set(
            pos,
            StringView(reinterpret_cast<const char*>(valueSrcPtr), length));
      } else {
        memcpy(rawBuffer, valueSrcPtr, length);
        column->setNoCopy(pos, StringView(rawBuffer, length));
        rawBuffer += length;
      }
    } else {
      column->setNull(pos, true);
    }
  }
  return column;
}

template <>
VectorPtr createFlatVectorFast<TypeKind::VARBINARY>(
    const TypePtr& type,
    int32_t columnIdx,
    int32_t numRows,
    int64_t fieldOffset,
    const std::vector<size_t>& offsets,
    const uint8_t* memoryAddress,
    memory::MemoryPool* pool) {
  return createFlatVectorFast<TypeKind::VARCHAR>(
      type, columnIdx, numRows, fieldOffset, offsets, memoryAddress, pool);
}

VectorPtr createUnknownFlatVector(int32_t numRows, memory::MemoryPool* pool) {
  auto nulls = allocateNulls(numRows, pool, bits::kNull);
  return std::make_shared<FlatVector<UnknownValue>>(
      pool,
      UNKNOWN(),
      nulls,
      numRows,
      nullptr, // values
      std::vector<BufferPtr>{}); // stringBuffers
}
} // namespace

// static
VectorPtr UnsafeRowDeserializer::deserialize(
    const uint8_t* memoryAddress,
    const RowTypePtr& type,
    const std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  auto numFields = type->size();
  int64_t nullBitsetWidthInBytes = UnsafeRow::getNullLength(numFields);
  std::vector<VectorPtr> columns(numFields);
  const vector_size_t numRows = offsets.size();

  for (auto i = 0; i < numFields; i++) {
    auto fieldOffset = getFieldOffset(nullBitsetWidthInBytes, i);
    auto& colType = type->childAt(i);
    if (colType->kind() == TypeKind::UNKNOWN) {
      columns[i] = createUnknownFlatVector(numRows, pool);
    } else {
      columns[i] = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createFlatVectorFast,
          colType->kind(),
          colType,
          i,
          numRows,
          fieldOffset,
          offsets,
          memoryAddress,
          pool);
    }
  }

  return std::make_shared<RowVector>(
      pool, type, BufferPtr(nullptr), numRows, std::move(columns));
}
} // namespace facebook::velox::row
