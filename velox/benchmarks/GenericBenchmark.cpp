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

#include <gflags/gflags.h>
#include "velox/benchmarks/QueryBenchmarkBase.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Task.h"
#include "velox/exec/TraceUtil.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"

DEFINE_string(plan, "", "Path to input json file of the velox plan.");

using namespace facebook::velox;

namespace {
core::PlanNodePtr getPlanNode(std::string planFile, memory::MemoryPool* pool) {
  auto fs = filesystems::getFileSystem(planFile, nullptr);
  auto obj = exec::trace::getTaskMetadata(planFile, fs);

  return ISerializable::deserialize<core::PlanNode>(obj, pool);
}

class GenericBenchmark : public QueryBenchmarkBase {
 public:
  void run() {
    folly::BenchmarkSuspender suspender;
    const auto plan = getPlanNode(FLAGS_plan, pool_.get());
    std::shared_ptr<exec::Task> task;
    suspender.dismiss();
    exec::test::AssertQueryBuilder(plan).runWithoutResults(task);
  }

 private:
  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{
      rootPool_->addLeafChild("GenericBenchmark")};
};

BENCHMARK(runBenchmark) {
  GenericBenchmark benchmark;
  benchmark.run();
}
} // namespace

int main(int argc, char* argv[]) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize({});
  folly::runBenchmarks();
  return 0;
}
