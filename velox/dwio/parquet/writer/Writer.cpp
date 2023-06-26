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

#include "velox/dwio/parquet/writer/Writer.h"
#include <arrow/c/bridge.h> // @manual
#include <arrow/table.h> // @manual
#include "velox/vector/arrow/Bridge.h"

namespace facebook::velox::parquet {

void Writer::write(const RowVectorPtr& data) {
  ArrowArray array;
  ArrowSchema schema;
  exportToArrow(data, array, &pool_);
  exportToArrow(data, schema);
  PARQUET_ASSIGN_OR_THROW(
      auto recordBatch, arrow::ImportRecordBatch(&array, &schema));
  auto table = arrow::Table::Make(
      recordBatch->schema(), recordBatch->columns(), data->size());
  if (!arrowWriter_) {
    stream_ = std::make_shared<DataBufferSink>(
        finalSink_.get(),
        pool_,
        queryCtx_->queryConfig().dataBufferGrowRatio());
    auto arrowProperties = ::parquet::ArrowWriterProperties::Builder().build();
    PARQUET_ASSIGN_OR_THROW(
        arrowWriter_,
        ::parquet::arrow::FileWriter::Open(
            *recordBatch->schema(),
            arrow::default_memory_pool(),
            stream_,
            properties_,
            arrowProperties));
  }

  PARQUET_THROW_NOT_OK(arrowWriter_->WriteTable(*table, 10000));

  if (queryCtx_->queryConfig().dataBufferGrowRatio() > 1) {
    flush(); // No performance drop on 1TB dataset.
  }
}

void Writer::flush() {
  PARQUET_THROW_NOT_OK(stream_->Flush());
}

void Writer::newRowGroup(int32_t numRows) {
  PARQUET_THROW_NOT_OK(arrowWriter_->NewRowGroup(numRows));
}

void Writer::close() {
  if (arrowWriter_) {
    PARQUET_THROW_NOT_OK(arrowWriter_->Close());
    arrowWriter_.reset();
  }
  PARQUET_THROW_NOT_OK(stream_->Close());
}

} // namespace facebook::velox::parquet
