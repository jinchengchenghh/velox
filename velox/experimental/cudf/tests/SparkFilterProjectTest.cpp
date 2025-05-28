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
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/sparksql/registration/Register.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

namespace {
class SparkCudfFilterProjectTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    // OperatorTestBase registers all the presto functons, the overwrite might
    // work well for decimal type argument functions, may need to refactor if
    // test decimal.
    functions::sparksql::registerFunctions("");
  }
};

TEST_F(SparkCudfFilterProjectTest, hashWithSeed) {
  auto input = makeRowVector(
      {makeConstant<int32_t>(42, 3),
       makeFlatVector<int64_t>(
           {-6041664978295882827, 42, 4904562767517797033})});
  auto vectors = {input};
  auto plan = PlanBuilder(pool_.get())
                  .values(vectors)
                  .project({"hash_with_seed(c0, c1) AS result"})
                  .planNode();
  auto expected =
      makeRowVector({makeFlatVector<int32_t>({-604166, 42, 490456})});
  // Run the test
  assertQuery(plan, expected);
}
} // namespace
