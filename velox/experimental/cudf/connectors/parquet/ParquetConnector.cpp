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

#include "velox/experimental/cudf/connectors/parquet/ParquetConfig.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetConnector.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSink.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetTableHandle.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

using namespace facebook::velox::connector;
using namespace facebook::velox::connector;
using namespace facebook::velox::config;
namespace facebook::velox::cudf_velox::connector::parquet {

namespace {
class ParquetConnector final : public Connector {
 public:
  ParquetConnector(
      const std::string& id,
      std::shared_ptr<const ConfigBase> config,
      folly::Executor* executor)
      : Connector(id),
        parquetConfig_(std::make_shared<ParquetConfig>(config)),
        executor_(executor) {
    LOG(INFO) << "cudf::Parquet connector " << connectorId() << " created.";
  }

  std::unique_ptr<DataSource> createDataSource(
      const std::shared_ptr<const RowType>& outputType,
      const std::shared_ptr<ConnectorTableHandle>& tableHandle,
      const std::unordered_map<std::string, std::shared_ptr<ColumnHandle>>&
          columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) override final {
    return std::make_unique<ParquetDataSource>(
        outputType,
        tableHandle,
        columnHandles,
        executor_,
        connectorQueryCtx,
        parquetConfig_);
  }

  const std::shared_ptr<const ConfigBase>& connectorConfig() const override {
    return parquetConfig_->config();
  }

  std::unique_ptr<DataSink> createDataSink(
      RowTypePtr inputType,
      std::shared_ptr<ConnectorInsertTableHandle> connectorInsertTableHandle,
      ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy) override final {
    auto parquetInsertHandle =
        std::dynamic_pointer_cast<ParquetInsertTableHandle>(
            connectorInsertTableHandle);
    VELOX_CHECK_NOT_NULL(
        parquetInsertHandle,
        "Parquet connector expecting parquet write handle!");
    return std::make_unique<ParquetDataSink>(
        inputType,
        parquetInsertHandle,
        connectorQueryCtx,
        CommitStrategy::kNoCommit,
        parquetConfig_);
  }

  folly::Executor* executor() const override {
    return executor_;
  }

 protected:
  const std::shared_ptr<ParquetConfig> parquetConfig_;
  folly::Executor* executor_;
};
} // namespace

std::shared_ptr<Connector> ParquetConnectorFactory::newConnector(
    const std::string& id,
    std::shared_ptr<const facebook::velox::config::ConfigBase> config,
    folly::Executor* ioExecutor,
    folly::Executor* cpuExecutor) {
  return std::make_shared<ParquetConnector>(id, config, ioExecutor);
}

} // namespace facebook::velox::cudf_velox::connector::parquet
