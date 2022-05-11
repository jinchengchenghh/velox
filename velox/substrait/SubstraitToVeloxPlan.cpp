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

#include "velox/substrait/SubstraitToVeloxPlan.h"
#include "velox/substrait/TypeUtils.h"
#include "velox/substrait/VariantToVectorConverter.h"
#include "velox/type/Type.h"

namespace facebook::velox::substrait {
namespace {
core::AggregationNode::Step toAggregationStep(
    const ::substrait::AggregateRel& sAgg) {
  if (sAgg.measures().size() == 0) {
    // When only groupings exist, set the phase to be Single.
    return core::AggregationNode::Step::kSingle;
  }

  // Use the first measure to set aggregation phase.
  const auto& firstMeasure = sAgg.measures()[0];
  const auto& aggFunction = firstMeasure.measure();
  switch (aggFunction.phase()) {
    case ::substrait::AGGREGATION_PHASE_INITIAL_TO_INTERMEDIATE:
      return core::AggregationNode::Step::kPartial;
    case ::substrait::AGGREGATION_PHASE_INTERMEDIATE_TO_INTERMEDIATE:
      return core::AggregationNode::Step::kIntermediate;
    case ::substrait::AGGREGATION_PHASE_INTERMEDIATE_TO_RESULT:
      return core::AggregationNode::Step::kFinal;
    case ::substrait::AGGREGATION_PHASE_INITIAL_TO_RESULT:
      return core::AggregationNode::Step::kSingle;
    default:
      VELOX_FAIL("Aggregate phase is not supported.");
  }
}
} // namespace

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::JoinRel& sJoin) {
  if (!sJoin.has_left()) {
    VELOX_FAIL("Left Rel is expected in JoinRel.");
  }
  if (!sJoin.has_right()) {
    VELOX_FAIL("Right Rel is expected in JoinRel.");
  }

  auto leftNode = toVeloxPlan(sJoin.left());
  auto rightNode = toVeloxPlan(sJoin.right());

  auto outputSize =
      leftNode->outputType()->size() + rightNode->outputType()->size();
  std::vector<std::string> outputNames;
  std::vector<std::shared_ptr<const Type>> outputTypes;
  outputNames.reserve(outputSize);
  outputTypes.reserve(outputSize);
  for (const auto& node : {leftNode, rightNode}) {
    const auto& names = node->outputType()->names();
    outputNames.insert(outputNames.end(), names.begin(), names.end());
    const auto& types = node->outputType()->children();
    outputTypes.insert(outputTypes.end(), types.begin(), types.end());
  }
  auto outputRowType = std::make_shared<const RowType>(
      std::move(outputNames), std::move(outputTypes));

  // extract join keys from join expression
  std::vector<const ::substrait::Expression::FieldReference*> leftExprs,
      rightExprs;
  extractJoinKeys(sJoin.expression(), leftExprs, rightExprs);
  VELOX_CHECK_EQ(leftExprs.size(), rightExprs.size());
  size_t numKeys = leftExprs.size();

  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> leftKeys,
      rightKeys;
  leftKeys.reserve(numKeys);
  rightKeys.reserve(numKeys);
  for (size_t i = 0; i < numKeys; ++i) {
    leftKeys.emplace_back(
        exprConverter_->toVeloxExpr(*leftExprs[i], outputRowType));
    rightKeys.emplace_back(
        exprConverter_->toVeloxExpr(*rightExprs[i], outputRowType));
  }

  std::shared_ptr<const core::ITypedExpr> filter;
  if (sJoin.has_post_join_filter()) {
    filter =
        exprConverter_->toVeloxExpr(sJoin.post_join_filter(), outputRowType);
  }

  // Map join type
  core::JoinType joinType;
  switch (sJoin.type()) {
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_INNER:
      joinType = core::JoinType::kInner;
      break;
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_OUTER:
      joinType = core::JoinType::kFull;
      break;
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_LEFT:
      joinType = core::JoinType::kLeft;
      break;
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_RIGHT:
      joinType = core::JoinType::kRight;
      break;
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_SEMI:
      joinType = core::JoinType::kLeftSemi;
      break;
    case ::substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_ANTI:
      joinType = core::JoinType::kNullAwareAnti;
      break;
    default:
      VELOX_NYI("Unsupported Join type: {}", sJoin.type());
  }

  // Create join node
  return std::make_shared<core::HashJoinNode>(
      nextPlanNodeId(),
      joinType,
      leftKeys,
      rightKeys,
      filter,
      leftNode,
      rightNode,
      outputRowType);
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::AggregateRel& sAgg) {
  std::shared_ptr<const core::PlanNode> childNode;
  if (sAgg.has_input()) {
    childNode = toVeloxPlan(sAgg.input());
  } else {
    VELOX_FAIL("Child Rel is expected in AggregateRel.");
  }
  core::AggregationNode::Step aggStep = toAggregationStep(sAgg);
  return toVeloxAgg(sAgg, childNode, aggStep);
}

std::shared_ptr<const core::PlanNode> SubstraitVeloxPlanConverter::toVeloxAgg(
    const ::substrait::AggregateRel& sAgg,
    const std::shared_ptr<const core::PlanNode>& childNode,
    const core::AggregationNode::Step& aggStep) {
  const auto& inputType = childNode->outputType();
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>
      veloxGroupingExprs;

  // Get the grouping expressions.
  uint32_t groupingOutIdx = 0;
  for (const auto& grouping : sAgg.groupings()) {
    for (const auto& groupingExpr : grouping.grouping_expressions()) {
      // Velox's groupings are limited to be Field.
      veloxGroupingExprs.emplace_back(
          exprConverter_->toVeloxExpr(groupingExpr.selection(), inputType));
      groupingOutIdx += 1;
    }
  }

  // Parse measures and get the aggregate expressions.
  // Each measure represents one aggregate expression.
  std::vector<std::shared_ptr<const core::CallTypedExpr>> aggExprs;
  aggExprs.reserve(sAgg.measures().size());
  for (const auto& smea : sAgg.measures()) {
    core::FieldAccessTypedExprPtr aggregateMask;
    ::substrait::Expression substraitAggMask = measure.filter();
    // Get Aggregation Masks.
    if (measure.has_filter()) {
      if (substraitAggMask.ByteSizeLong() == 0) {
        aggregateMask = {};
      } else {
        aggregateMask =
            std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                exprConverter_->toVeloxExpr(substraitAggMask, inputType));
      }
      aggregateMasks.push_back(aggregateMask);
    }
    const auto& aggFunction = smea.measure();
    std::string funcName = subParser_->findVeloxFunction(
        functionMap_, aggFunction.function_reference());
    std::vector<std::shared_ptr<const core::ITypedExpr>> aggParams;
    aggParams.reserve(aggFunction.arguments().size());
    for (const auto& arg : aggFunction.arguments()) {
      aggParams.emplace_back(exprConverter_->toVeloxExpr(
          getExprFromFunctionArgument(arg), inputType));
    }
    auto aggVeloxType =
        toVeloxType(subParser_->parseType(aggFunction.output_type())->type);
    auto aggExpr = std::make_shared<const core::CallTypedExpr>(
        aggVeloxType, std::move(aggParams), funcName);
    aggExprs.emplace_back(aggExpr);
  }

  bool ignoreNullKeys = false;
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> aggregateMasks(
      sAgg.measures().size());
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>
      preGroupingExprs = {};

  // Get the output names of Aggregation.
  std::vector<std::string> aggOutNames;
  aggOutNames.reserve(sAgg.measures().size());
  for (int idx = groupingOutIdx; idx < groupingOutIdx + sAgg.measures().size();
       idx++) {
    aggOutNames.emplace_back(subParser_->makeNodeName(planNodeId_, idx));
  }

  // Create Aggregate node.
  auto aggNode = std::make_shared<core::AggregationNode>(
      nextPlanNodeId(),
      aggStep,
      veloxGroupingExprs,
      preGroupingExprs,
      aggOutNames,
      aggExprs,
      aggregateMasks,
      ignoreNullKeys,
      childNode);
  return aggNode;
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::ProjectRel& projectRel) {
  core::PlanNodePtr childNode;
  if (projectRel.has_input()) {
    childNode = toVeloxPlan(projectRel.input());
  } else {
    VELOX_FAIL("Child Rel is expected in ProjectRel.");
  }
}
core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::ProjectRel& projectRel,
    memory::MemoryPool* pool) {
  core::PlanNodePtr childNode;
  if (projectRel.has_input()) {
    childNode = toVeloxPlan(projectRel.input(), pool);
  } else {
    VELOX_FAIL("Child Rel is expected in ProjectRel.");
  }

  // Construct Velox Expressions.
  const auto& projectExprs = sProject.expressions();
  std::vector<std::string> projectNames;
  std::vector<core::TypedExprPtr> expressions;
  projectNames.reserve(projectExprs.size());
  expressions.reserve(projectExprs.size());

  const auto& inputType = childNode->outputType();
  int colIdx = 0;
  for (const auto& expr : projectExprs) {
    expressions.emplace_back(exprConverter_->toVeloxExpr(expr, inputType));
    projectNames.emplace_back(
        substraitParser_->makeNodeName(planNodeId_, colIdx));
    colIdx += 1;
  }

  return std::make_shared<core::ProjectNode>(
      nextPlanNodeId(),
      std::move(projectNames),
      std::move(expressions),
      childNode);
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::FilterRel& filterRel) {
  core::PlanNodePtr childNode;
  if (filterRel.has_input()) {
    childNode = toVeloxPlan(filterRel.input());
  } else {
    VELOX_FAIL("Child Rel is expected in FilterRel.");
  }

  const auto& inputType = childNode->outputType();
  const auto& sExpr = filterRel.condition();

  return std::make_shared<core::FilterNode>(
      nextPlanNodeId(),
      exprConverter_->toVeloxExpr(sExpr, inputType),
      childNode);
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::ReadRel& sRead) {
  // Check if the ReadRel specifies an input of stream. If yes, the pre-built
  // input node will be used as the data source.
  auto splitInfo = std::make_shared<SplitInfo>();
  auto streamIdx = streamIsInput(sRead);
  if (streamIdx >= 0) {
    if (inputNodesMap_.find(streamIdx) == inputNodesMap_.end()) {
      VELOX_FAIL(
          "Could not find source index {} in input nodes map.", streamIdx);
    }
    auto streamNode = inputNodesMap_[streamIdx];
    splitInfo->isStream = true;
    splitInfoMap_[streamNode->id()] = splitInfo;
    return streamNode;
  }

  // Otherwise, will create TableScan node for ReadRel.
  // Get output names and types.
  std::vector<std::string> colNameList;
  std::vector<TypePtr> veloxTypeList;
  if (sRead.has_base_schema()) {
    const auto& baseSchema = sRead.base_schema();
    colNameList.reserve(baseSchema.names().size());
    for (const auto& name : baseSchema.names()) {
      colNameList.emplace_back(name);
    }
    auto substraitTypeList = subParser_->parseNamedStruct(baseSchema);
    veloxTypeList.reserve(substraitTypeList.size());
    for (const auto& substraitType : substraitTypeList) {
      veloxTypeList.emplace_back(toVeloxType(substraitType->type));
    }
  }

  // Parse local files and construct split info.
  if (sRead.has_local_files()) {
    using SubstraitFileFormatCase =
        ::substrait::ReadRel_LocalFiles_FileOrFiles::FileFormatCase;
    const auto& fileList = sRead.local_files().items();
    splitInfo->paths.reserve(fileList.size());
    splitInfo->starts.reserve(fileList.size());
    splitInfo->lengths.reserve(fileList.size());
    for (const auto& file : fileList) {
      // Expect all Partitions share the same index.
      splitInfo->partitionIndex = file.partition_index();
      splitInfo->paths.emplace_back(file.uri_file());
      splitInfo->starts.emplace_back(file.start());
      splitInfo->lengths.emplace_back(file.length());
      switch (file.file_format_case()) {
        case SubstraitFileFormatCase::kOrc:
        case SubstraitFileFormatCase::kDwrf:
          splitInfo->format = dwio::common::FileFormat::DWRF;
          break;
        case SubstraitFileFormatCase::kParquet:
          splitInfo->format = dwio::common::FileFormat::PARQUET;
          break;
        default:
          splitInfo->format = dwio::common::FileFormat::UNKNOWN;
      }
    }
  }
  // Do not hard-code connector ID and allow for connectors other than Hive.
  static const std::string kHiveConnectorId = "test-hive";

  // Velox requires Filter Pushdown must being enabled.
  bool filterPushdownEnabled = true;
  std::shared_ptr<connector::hive::HiveTableHandle> tableHandle;
  if (!sRead.has_filter()) {
    tableHandle = std::make_shared<connector::hive::HiveTableHandle>(
        kHiveConnectorId,
        "hive_table",
        filterPushdownEnabled,
        connector::hive::SubfieldFilters{},
        nullptr);
  } else {
    // Flatten the conditions connected with 'and'.
    std::vector<::substrait::Expression_ScalarFunction> scalarFunctions;
    std::vector<::substrait::Expression_SingularOrList> singularOrLists;
    flattenConditions(sRead.filter(), scalarFunctions, singularOrLists);

    std::unordered_map<uint32_t, std::shared_ptr<RangeRecorder>> rangeRecorders;
    for (uint32_t idx = 0; idx < veloxTypeList.size(); idx++) {
      rangeRecorders[idx] = std::make_shared<RangeRecorder>();
    }

    // Separate the filters to be two parts. The subfield part can be
    // pushed down.
    std::vector<::substrait::Expression_ScalarFunction> subfieldFunctions;
    std::vector<::substrait::Expression_SingularOrList> subfieldrOrLists;

    std::vector<::substrait::Expression_ScalarFunction> remainingFunctions;
    std::vector<::substrait::Expression_SingularOrList> remainingrOrLists;

    separateFilters(
        rangeRecorders,
        scalarFunctions,
        subfieldFunctions,
        remainingFunctions,
        singularOrLists,
        subfieldrOrLists,
        remainingrOrLists);

    // Create subfield filters based on the constructed filter info map.
    connector::hive::SubfieldFilters subfieldFilters = toSubfieldFilters(
        colNameList, veloxTypeList, subfieldFunctions, subfieldrOrLists);
    // Connect the remaining filters with 'and'.
    std::shared_ptr<const core::ITypedExpr> remainingFilter;

    if (!isPushDownSupportedByFormat(splitInfo->format, subfieldFilters)) {
      // A subfieldFilter is not supported by the format,
      // mark all filter as remaining filters.
      subfieldFilters.clear();
      remainingFilter = connectWithAnd(
          colNameList, veloxTypeList, scalarFunctions, singularOrLists);
    } else {
      remainingFilter = connectWithAnd(
          colNameList, veloxTypeList, remainingFunctions, remainingrOrLists);
    }

    tableHandle = std::make_shared<connector::hive::HiveTableHandle>(
        kHiveConnectorId,
        "hive_table",
        filterPushdownEnabled,
        std::move(subfieldFilters),
        remainingFilter);
  }

  // Get assignments and out names.
  std::vector<std::string> outNames;
  outNames.reserve(colNameList.size());
  std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
      assignments;
  for (int idx = 0; idx < colNameList.size(); idx++) {
    auto outName = subParser_->makeNodeName(planNodeId_, idx);
    assignments[outName] = std::make_shared<connector::hive::HiveColumnHandle>(
        colNameList[idx],
        connector::hive::HiveColumnHandle::ColumnType::kRegular,
        veloxTypeList[idx]);
    outNames.emplace_back(outName);
  }
  auto outputType = ROW(std::move(outNames), std::move(veloxTypeList));

  if (sRead.has_virtual_table()) {
    return toVeloxPlan(sRead, outputType);
  } else {
    auto tableScanNode = std::make_shared<core::TableScanNode>(
        nextPlanNodeId(), outputType, tableHandle, assignments);
    // Set split info map.
    splitInfoMap_[tableScanNode->id()] = splitInfo;
    return tableScanNode;
  }
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::ReadRel& readRel,
    const RowTypePtr& type) {
  ::substrait::ReadRel_VirtualTable readVirtualTable = readRel.virtual_table();
  int64_t numVectors = readVirtualTable.values_size();
  int64_t numColumns = type->size();
  int64_t valueFieldNums =
      readVirtualTable.values(numVectors - 1).fields_size();
  std::vector<RowVectorPtr> vectors;
  vectors.reserve(numVectors);

  int64_t batchSize = valueFieldNums / numColumns;

  for (int64_t index = 0; index < numVectors; ++index) {
    std::vector<VectorPtr> children;
    ::substrait::Expression_Literal_Struct rowValue =
        readRel.virtual_table().values(index);
    auto fieldSize = rowValue.fields_size();
    VELOX_CHECK_EQ(fieldSize, batchSize * numColumns);

    for (int64_t col = 0; col < numColumns; ++col) {
      const TypePtr& outputChildType = type->childAt(col);
      std::vector<variant> batchChild;
      batchChild.reserve(batchSize);
      for (int64_t batchId = 0; batchId < batchSize; batchId++) {
        // each value in the batch
        auto fieldIdx = col * batchSize + batchId;
        ::substrait::Expression_Literal field = rowValue.fields(fieldIdx);

        auto expr = exprConverter_->toVeloxExpr(field);
        if (auto constantExpr =
                std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
                    expr)) {
          if (!constantExpr->hasValueVector()) {
            batchChild.emplace_back(constantExpr->value());
          } else {
            VELOX_UNSUPPORTED(
                "Values node with complex type values is not supported yet");
          }
        } else {
          VELOX_FAIL("Expected constant expression");
        }
      }
      children.emplace_back(
          setVectorFromVariants(outputChildType, batchChild, pool_));
    }

    vectors.emplace_back(
        std::make_shared<RowVector>(pool_, type, nullptr, batchSize, children));
  }
  return std::make_shared<core::ValuesNode>(nextPlanNodeId(), vectors);
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::Rel& sRel) {
  if (sRel.has_aggregate()) {
    return toVeloxPlan(sRel.aggregate());
  }
  if (sRel.has_project()) {
    return toVeloxPlan(sRel.project());
  }
  if (sRel.has_filter()) {
    return toVeloxPlan(sRel.filter());
  }
  if (sRel.has_join()) {
    return toVeloxPlan(sRel.join());
  }
  if (sRel.has_read()) {
    return toVeloxPlan(sRel.read());
  }
  if (sRel.has_sort()) {
    return toVeloxPlan(sRel.sort());
  }
  VELOX_NYI("Substrait conversion not supported for Rel.");
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::RelRoot& root) {
  // TODO: Use the names as the output names for the whole computing.
  const auto& names = root.names();
  if (root.has_input()) {
    const auto& rel = root.input();
    return toVeloxPlan(rel);
  }
  VELOX_FAIL("Input is expected in RelRoot.");
}

core::PlanNodePtr SubstraitVeloxPlanConverter::toVeloxPlan(
    const ::substrait::Plan& substraitPlan) {
  VELOX_CHECK(
      checkTypeExtension(substraitPlan),
      "The type extension only have unknown type.")
  // Construct the function map based on the Substrait representation,
  // and initialize the expression converter with it.
  constructFunctionMap(substraitPlan);

  // In fact, only one RelRoot or Rel is expected here.
  VELOX_CHECK_EQ(substraitPlan.relations_size(), 1);
  const auto& sRel = substraitPlan.relations(0);
  if (sRel.has_root()) {
    return toVeloxPlan(sRel.root());
  }
  if (sRel.has_rel()) {
    return toVeloxPlan(sRel.rel());
  }

  VELOX_FAIL("RelRoot or Rel is expected in Plan.");
}

void SubstraitVeloxPlanConverter::constructFunctionMap(
    const ::substrait::Plan& substraitPlan) {
  // Construct the function map based on the Substrait representation.
  for (const auto& sExtension : substraitPlan.extensions()) {
    if (!sExtension.has_extension_function()) {
      continue;
    }
    const auto& sFmap = sExtension.extension_function();
    auto id = sFmap.function_anchor();
    auto name = sFmap.name();
    functionMap_[id] = name;
  }
  exprConverter_ =
      std::make_shared<SubstraitVeloxExprConverter>(pool_, functionMap_);
}

std::string SubstraitVeloxPlanConverter::nextPlanNodeId() {
  auto id = fmt::format("{}", planNodeId_);
  planNodeId_++;
  return id;
}

// This class contains the needed infos for Filter Pushdown.
// TODO: Support different types here.
class FilterInfo {
 public:
  // Used to set the left bound.
  void setLeft(double left, bool isExclusive) {
    left_ = left;
    leftExclusive_ = isExclusive;
    if (!isInitialized_) {
      isInitialized_ = true;
    }
  }

  // Used to set the right bound.
  void setRight(double right, bool isExclusive) {
    right_ = right;
    rightExclusive_ = isExclusive;
    if (!isInitialized_) {
      isInitialized_ = true;
    }
  }

  // Will fordis Null value if called once.
  void forbidsNull() {
    nullAllowed_ = false;
    if (!isInitialized_) {
      isInitialized_ = true;
    }
  }

  // Return the initialization status.
  bool isInitialized() {
    return isInitialized_ ? true : false;
  }

  // The left bound.
  std::optional<double> left_ = std::nullopt;
  // The right bound.
  std::optional<double> right_ = std::nullopt;
  // The Null allowing.
  bool nullAllowed_ = true;
  // If true, left bound will be exclusive.
  bool leftExclusive_ = false;
  // If true, right bound will be exclusive.
  bool rightExclusive_ = false;

 private:
  bool isInitialized_ = false;
};

connector::hive::SubfieldFilters SubstraitVeloxPlanConverter::toVeloxFilter(
    const std::vector<std::string>& inputNameList,
    const std::vector<TypePtr>& inputTypeList,
    const ::substrait::Expression& substraitFilter) {
  connector::hive::SubfieldFilters filters;
  // A map between the column index and the FilterInfo for that column.
  std::unordered_map<int, std::shared_ptr<FilterInfo>> colInfoMap;
  for (int idx = 0; idx < inputNameList.size(); idx++) {
    colInfoMap[idx] = std::make_shared<FilterInfo>();
  }

  std::vector<::substrait::Expression_ScalarFunction> scalarFunctions;
  flattenConditions(substraitFilter, scalarFunctions);
  // Construct the FilterInfo for the related column.
  for (const auto& scalarFunction : scalarFunctions) {
    auto filterNameSpec = substraitParser_->findFunctionSpec(
        functionMap_, scalarFunction.function_reference());
    auto filterName = substraitParser_->getFunctionName(filterNameSpec);
    int32_t colIdx;
    // TODO: Add different types' support here.
    double val;
    for (auto& param : scalarFunction.args()) {
      auto typeCase = param.rex_type_case();
      switch (typeCase) {
        case ::substrait::Expression::RexTypeCase::kSelection: {
          auto sel = param.selection();
          // TODO: Only direct reference is considered here.
          auto dRef = sel.direct_reference();
          colIdx = substraitParser_->parseReferenceSegment(dRef);
          break;
        }
        case ::substrait::Expression::RexTypeCase::kLiteral: {
          auto sLit = param.literal();
          // TODO: Only double is considered here.
          val = sLit.fp64();
          break;
        }
        default:
          VELOX_NYI(
              "Substrait conversion not supported for arg type '{}'", typeCase);
      }
    }
    if (filterName == "is_not_null") {
      colInfoMap[colIdx]->forbidsNull();
    } else if (filterName == "gte") {
      colInfoMap[colIdx]->setLeft(val, false);
    } else if (filterName == "gt") {
      colInfoMap[colIdx]->setLeft(val, true);
    } else if (filterName == "lte") {
      colInfoMap[colIdx]->setRight(val, false);
    } else if (filterName == "lt") {
      colInfoMap[colIdx]->setRight(val, true);
    } else {
      VELOX_NYI(
          "Substrait conversion not supported for filter name '{}'",
          filterName);
    }
  }

  // Construct the Filters.
  for (int idx = 0; idx < inputNameList.size(); idx++) {
    auto filterInfo = colInfoMap[idx];
    // Set the left bound to be negative infinity.
    double leftBound = -1.0 / 0.0;
    // Set the right bound to be positive infinity.
    double rightBound = 1.0 / 0.0;
    bool leftUnbounded = true;
    bool rightUnbounded = true;
    bool leftExclusive = false;
    bool rightExclusive = false;
    if (filterInfo->isInitialized()) {
      if (filterInfo->left_) {
        leftUnbounded = false;
        leftBound = filterInfo->left_.value();
        leftExclusive = filterInfo->leftExclusive_;
      }
      if (filterInfo->right_) {
        rightUnbounded = false;
        rightBound = filterInfo->right_.value();
        rightExclusive = filterInfo->rightExclusive_;
      }
      bool nullAllowed = filterInfo->nullAllowed_;
      filters[common::Subfield(inputNameList[idx])] =
          std::make_unique<common::DoubleRange>(
              leftBound,
              leftUnbounded,
              leftExclusive,
              rightBound,
              rightUnbounded,
              rightExclusive,
              nullAllowed);
    }
  }
  return filters;
}

void SubstraitVeloxPlanConverter::flattenConditions(
    const ::substrait::Expression& substraitFilter,
    std::vector<::substrait::Expression_ScalarFunction>& scalarFunctions) {
  auto typeCase = substraitFilter.rex_type_case();
  switch (typeCase) {
    case ::substrait::Expression::RexTypeCase::kScalarFunction: {
      auto sFunc = substraitFilter.scalar_function();
      auto filterNameSpec = substraitParser_->findFunctionSpec(
          functionMap_, sFunc.function_reference());
      // TODO: Only and relation is supported here.
      if (substraitParser_->getFunctionName(filterNameSpec) == "and") {
        for (const auto& sCondition : sFunc.args()) {
          flattenConditions(sCondition, scalarFunctions);
        }
      } else {
        scalarFunctions.emplace_back(sFunc);
      }
      break;
    }
    default:
      VELOX_NYI("GetFlatConditions not supported for type '{}'", typeCase);
  }
}

std::string SubstraitVeloxPlanConverter::findFuncSpec(uint64_t id) {
  return subParser_->findSubstraitFuncSpec(functionMap_, id);
}

int32_t SubstraitVeloxPlanConverter::streamIsInput(
    const ::substrait::ReadRel& sRead) {
  if (sRead.has_local_files()) {
    const auto& fileList = sRead.local_files().items();
    if (fileList.size() == 0) {
      VELOX_FAIL("At least one file path is expected.");
    }

    // The stream input will be specified with the format of
    // "iterator:${index}".
    std::string filePath = fileList[0].uri_file();
    std::string prefix = "iterator:";
    std::size_t pos = filePath.find(prefix);
    if (pos == std::string::npos) {
      return -1;
    }

    // Get the index.
    std::string idxStr = filePath.substr(pos + prefix.size(), filePath.size());
    try {
      return stoi(idxStr);
    } catch (const std::exception& err) {
      VELOX_FAIL(err.what());
    }
  }
  if (validationMode_) {
    return -1;
  }
  VELOX_FAIL("Local file is expected.");
}

void SubstraitVeloxPlanConverter::flattenConditions(
    const ::substrait::Expression& substraitFilter,
    std::vector<::substrait::Expression_ScalarFunction>& scalarFunctions,
    std::vector<::substrait::Expression_SingularOrList>& singularOrLists) {
  auto typeCase = substraitFilter.rex_type_case();
  switch (typeCase) {
    case ::substrait::Expression::RexTypeCase::kScalarFunction: {
      auto sFunc = substraitFilter.scalar_function();
      auto filterNameSpec = subParser_->findSubstraitFuncSpec(
          functionMap_, sFunc.function_reference());
      // TODO: Only and relation is supported here.
      if (subParser_->getSubFunctionName(filterNameSpec) == "and") {
        for (const auto& sCondition : sFunc.arguments()) {
          flattenConditions(
              getExprFromFunctionArgument(sCondition),
              scalarFunctions,
              singularOrLists);
        }
      } else {
        scalarFunctions.emplace_back(sFunc);
      }
      break;
    }
    case ::substrait::Expression::RexTypeCase::kSingularOrList: {
      singularOrLists.emplace_back(substraitFilter.singular_or_list());
      break;
    }
    default:
      VELOX_NYI("GetFlatConditions not supported for type '{}'", typeCase);
  }
}

void SubstraitVeloxPlanConverter::constructFunctionMap(
    const ::substrait::Plan& substraitPlan) {
  // Construct the function map based on the Substrait representation.
  for (const auto& sExtension : substraitPlan.extensions()) {
    if (!sExtension.has_extension_function()) {
      continue;
    }
    const auto& sFmap = sExtension.extension_function();
    auto id = sFmap.function_anchor();
    auto name = sFmap.name();
    functionMap_[id] = name;
  }
}

bool SubstraitVeloxPlanConverter::checkTypeExtension(
    const ::substrait::Plan& substraitPlan) {
  for (const auto& sExtension : substraitPlan.extensions()) {
    if (!sExtension.has_extension_type()) {
      continue;
    }

    // Only support UNKNOWN type in UserDefined type extension.
    if (sExtension.extension_type().name() != "UNKNOWN") {
      return false;
    }
  }
  return true;
}

const std::string& SubstraitVeloxPlanConverter::findFunction(
    uint64_t id) const {
  return substraitParser_->findFunctionSpec(functionMap_, id);
}

void SubstraitVeloxPlanConverter::extractJoinKeys(
    const ::substrait::Expression& joinExpression,
    std::vector<const ::substrait::Expression::FieldReference*>& leftExprs,
    std::vector<const ::substrait::Expression::FieldReference*>& rightExprs) {
  std::vector<const ::substrait::Expression*> expressions;
  expressions.push_back(&joinExpression);
  while (!expressions.empty()) {
    auto visited = expressions.back();
    expressions.pop_back();
    if (visited->rex_type_case() ==
        ::substrait::Expression::RexTypeCase::kScalarFunction) {
      const auto& funcName =
          subParser_->getSubFunctionName(subParser_->findVeloxFunction(
              functionMap_, visited->scalar_function().function_reference()));
      const auto& args = visited->scalar_function().args();
      if (funcName == "and") {
        expressions.push_back(&args[0]);
        expressions.push_back(&args[1]);
      } else if (funcName == "equal") {
        VELOX_CHECK(std::all_of(
            args.cbegin(), args.cend(), [](const ::substrait::Expression& arg) {
              return arg.has_selection();
            }));
        leftExprs.push_back(&args[0].selection());
        rightExprs.push_back(&args[1].selection());
      }
    } else {
      VELOX_FAIL(
          "Unable to parse from join expression: {}",
          joinExpression.DebugString());
    }
  }
}

} // namespace facebook::velox::substrait
