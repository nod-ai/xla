#include "xla/xla_cc.h"

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mhlo/IR/register.h>
#include <mlir/Bytecode/BytecodeWriter.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <stablehlo/dialect/Register.h>

#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>

#include "hlo/experimental/auto_sharding/auto_sharding.h"
#include "tsl/platform/errors.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/all_gather_broadcast_reorder.h"
#include "xla/service/all_reduce_folder.h"
#include "xla/service/all_reduce_reassociate.h"
#include "xla/service/call_inliner.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/conditional_canonicalizer.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/gpu/gpu_reduce_scatter_creator.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/reduce_scatter_reassociate.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/spmd/collective_permute_motion.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "xla/service/while_loop_all_reduce_code_motion.h"
#include "xla/service/zero_sized_hlo_elimination.h"
#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla_cc.h"

using namespace xla;

namespace {

XlaStatus fromXlaStatus(xla::Status s) {
  if (s == OkStatus()) {
    return XlaStatus::OK;
  }

  return XlaStatus::ERROR;
}

std::unique_ptr<mlir::MLIRContext> makeMlirContext() {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  return context;
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> loadMlir(
    const char* mlirFilePath, mlir::MLIRContext& context) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(mlirFilePath);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file \"" << mlirFilePath
                 << "\": " << ec.message() << "\n";
    return mlir::failure();
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> loadMlir(
    const char* buffer, size_t size, mlir::MLIRContext& context) {
  std::unique_ptr<llvm::MemoryBuffer> memoryBuffer =
      llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(buffer, size),
                                       /*BufferName=*/"",
                                       /*RequiresNullTerminator=*/false);
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
}

XlaStatus stableHloToXlaHloPjrt(mlir::ModuleOp moduleOp,
                                xla::HloModule** outModule) {
  xla::XlaComputation xlaComputation;
  auto status = xla::MlirToXlaComputation(moduleOp, xlaComputation,
                                          /*use_tuple_args=*/false,
                                          /*return_tuple=*/false);
  if (!status.ok()) {
    std::cerr << "Failed to convert MLIR to XLA HLO: " << status.ToString()
              << std::endl;
    return XlaStatus::ERROR;
  }

  xla::DebugOptions debugOptions;
  xla::StatusOr<xla::HloModuleConfig> hloModuleConfig =
      xla::HloModule::CreateModuleConfigFromProto(xlaComputation.proto(),
                                                  debugOptions);
  if (!hloModuleConfig.ok()) {
    return XlaStatus::ERROR;
  }
  xla::StatusOr<std::unique_ptr<xla::HloModule>> hloModule =
      xla::HloModule::CreateFromProto(xlaComputation.proto(),
                                      hloModuleConfig.value());
  if (!hloModule.ok()) {
    return XlaStatus::ERROR;
  }

  if (auto num_partitions =
          moduleOp->getAttrOfType<mlir::IntegerAttr>("mhlo.num_partitions")) {
    hloModule.value()->config().set_num_partitions(num_partitions.getInt());
  }

  if (auto num_replicas =
          moduleOp->getAttrOfType<mlir::IntegerAttr>("mhlo.num_replicas")) {
    hloModule.value()->config().set_replica_count(num_replicas.getInt());
  }

  *outModule = hloModule.value().release();

  return XlaStatus::OK;
}

xla::StatusOr<mlir::ModuleOp> xlaHloToStableHlo(const xla::HloModule& module,
                                                mlir::MLIRContext& context) {
  mlir::ModuleOp moduleOp = mlir::OpBuilder(&context).create<mlir::ModuleOp>(
      mlir::UnknownLoc::get(&context));
  xla::Status status =
      xla::ConvertHloToMlirHlo(moduleOp, const_cast<xla::HloModule*>(&module));
  if (!status.ok()) {
    return status;
  }
  mlir::PassManager pm(&context);
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (!mlir::succeeded(pm.run(moduleOp))) {
    return tsl::errors::InvalidArgument("MHLO => StableHLO failed.");
  }
  return moduleOp;
}

int64_t shapeElementsCount(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), 1,
                         [](int64_t prod, int64_t x) { return prod * x; });
}

xla::AutoShardingOption::PreserveShardingsType
convertAutoShardingOptionPreserveShardingsType(
    XlaAutoShardingOptionPreserveShardingsType x) {
  switch (x) {
    case kKeepAllShardings:
      return xla::AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
    case kKeepInputOutputShardings:
      return xla::AutoShardingOption::PreserveShardingsType::
          kKeepInputOutputShardings;
    case kRemoveAllShardings:
      return xla::AutoShardingOption::PreserveShardingsType::
          kRemoveAllShardings;
  }
}

xla::AutoShardingOption convertAutoShardingOption(
    const XlaAutoShardingOption& option) {
  xla::AutoShardingOption res;
  res.enable = option.enable;
  res.preserve_shardings =
      convertAutoShardingOptionPreserveShardingsType(option.preserve_shardings);
  res.simplify_graph = option.simplify_graph;
  res.memory_budget_per_device = option.memory_budget_per_device;
  res.memory_budget_ratio = option.memory_budget_ratio;
  res.force_all_gather_cost = option.force_all_gather_cost;
  res.all_gather_cost = option.all_gather_cost;
  res.force_all_to_all_cost = option.force_all_to_all_cost;
  res.all_to_all_cost = option.all_to_all_cost;
  res.force_batch_dim_to_mesh_dim = option.force_batch_dim_to_mesh_dim;
  res.allow_replicated_parameters = option.allow_replicated_parameters;
  res.prefer_reduce_scatter = option.prefer_reduce_scatter;
  res.reduce_scatter_grad_acc_friendly =
      option.reduce_scatter_grad_acc_friendly;
  res.reduce_scatter_aggressive_partition =
      option.reduce_scatter_aggressive_partition;
  res.batch_matmul_always_split_batch = option.batch_matmul_always_split_batch;
  res.allow_recompute_heavy_op = option.allow_recompute_heavy_op;
  res.allow_mixed_mesh_shape = option.allow_mixed_mesh_shape;
  res.grad_acc_num_micro_batches = option.grad_acc_num_micro_batches;
  res.load_solution_vector = option.load_solution_vector;
  res.solve_nd_sharding_iteratively = option.solve_nd_sharding_iteratively;
  res.force_simple_heuristic = option.force_simple_heuristic;
  res.force_strategy = option.force_strategy;
  res.force_strategy_inst_indices = std::vector<int64_t>(
      option.force_strategy_inst_indices,
      option.force_strategy_inst_indices + option.force_strategy_size);
  std::transform(option.force_strategy_stra_names,
                 option.force_strategy_stra_names + option.force_strategy_size,
                 std::back_inserter(res.force_strategy_stra_names),
                 [](const char* x) { return std::string(x); });
  res.device_mesh_shape = std::vector<int64_t>(
      option.device_mesh_shape,
      option.device_mesh_shape + option.device_mesh_shape_size);
  int64_t deviceCount = shapeElementsCount(res.device_mesh_shape);
  res.device_mesh_ids = std::vector<int64_t>(
      option.device_mesh_ids, option.device_mesh_ids + deviceCount);
  res.device_mesh_alpha = std::vector<double>(
      option.device_mesh_alpha,
      option.device_mesh_alpha + option.device_mesh_shape_size);
  res.device_mesh_beta = std::vector<double>(
      option.device_mesh_beta,
      option.device_mesh_beta + option.device_mesh_shape_size);
  res.load_strategy = option.load_strategy;
  res.try_multiple_mesh_shapes = option.try_multiple_mesh_shapes;
  res.solver_timeout_in_seconds = option.solver_timeout_in_seconds;
  res.strategy_vector = std::vector<int64_t>(
      option.strategy_vector,
      option.strategy_vector + option.strategy_vector_size);
  return res;
}

}  // namespace

extern "C" {

XlaStatus xlaStableHloFileToXlaHlo(const char* filepath,
                                   xla::HloModule** outModule) {
  auto context = makeMlirContext();
  auto moduleOp = loadMlir(filepath, *context);
  if (failed(moduleOp)) {
    std::cerr << "Failed to load MLIR." << std::endl;
    return XlaStatus::ERROR;
  }
  return stableHloToXlaHloPjrt(*moduleOp.value(), outModule);
}

XlaStatus xlaStableHloBufferToXlaHlo(const char* mlirBytecodeBuffer,
                                     size_t mlirBytecodeBufferSize,
                                     xla::HloModule** outModule) {
  auto context = makeMlirContext();
  auto moduleOp =
      loadMlir(mlirBytecodeBuffer, mlirBytecodeBufferSize, *context);
  if (failed(moduleOp)) {
    std::cerr << "Failed to load MLIR." << std::endl;
    return XlaStatus::ERROR;
  }
  return stableHloToXlaHloPjrt(*moduleOp.value(), outModule);
}

void xlaDestroyHloModule(xla::HloModule* hloModule) {
  std::unique_ptr<xla::HloModule> ptr(hloModule);
}

void xlaDestroyCharBuffer(char* buff) { std::unique_ptr<char[]> ptr(buff); }

XlaStatus xlaXlaHloToStableHloBuffer(const xla::HloModule& module,
                                     char** outMlirBytecodeBuffer,
                                     size_t* outMlirBytecodeBufferSize) {
  auto context = makeMlirContext();
  xla::StatusOr<mlir::ModuleOp> moduleOp = xlaHloToStableHlo(module, *context);
  if (!moduleOp.ok()) {
    std::cout << "Failed converting XLA HLO to MLIR StableHLO: "
              << moduleOp.status() << std::endl;
    return XlaStatus::ERROR;
  }
  std::string rawBytecodeBuffer;
  llvm::raw_string_ostream os(rawBytecodeBuffer);
  if (failed(mlir::writeBytecodeToFile(moduleOp.value().getOperation(), os))) {
    std::cout << "Failed serializing MLIR to bytecode." << std::endl;
    return XlaStatus::ERROR;
  }
  auto result = std::make_unique<char[]>(rawBytecodeBuffer.size());
  *outMlirBytecodeBuffer = result.get();
  *outMlirBytecodeBufferSize = rawBytecodeBuffer.size();
  memcpy(*outMlirBytecodeBuffer, rawBytecodeBuffer.data(),
         *outMlirBytecodeBufferSize);
  result.release();
  return XlaStatus::OK;
}

void xlaMakeDefaultShardingPropagationOption(
    XlaShardingPropagationOption* option) {
  option->is_spmd = false;
  option->propagate_metadata = false;
  option->allow_spmd_sharding_propagation_to_output = nullptr;
  option->allow_spmd_sharding_propagation_to_output_size = 0;
  option->allow_spmd_sharding_propagation_to_parameters = nullptr;
  option->allow_spmd_sharding_propagation_to_parameters_size = 0;
  option->cse_prevention_only = false;
}

XlaStatus xlaRunShardingPropagationPass(
    xla::HloModule* module, const XlaShardingPropagationOption* option) {
  HloPassPipeline pipeline("sharding-propagation");
  absl::Span<const bool> allow_spmd_sharding_propagation_to_output(
      option->allow_spmd_sharding_propagation_to_output,
      option->allow_spmd_sharding_propagation_to_output_size);
  absl::Span<const bool> allow_spmd_sharding_propagation_to_parameters(
      option->allow_spmd_sharding_propagation_to_parameters,
      option->allow_spmd_sharding_propagation_to_parameters_size);
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<ZeroSizedHloElimination>();
  pipeline.AddPass<ConditionalCanonicalizer>();
  pipeline.AddPass<ShardingPropagation>(
      option->is_spmd, option->propagate_metadata,
      allow_spmd_sharding_propagation_to_output,
      allow_spmd_sharding_propagation_to_parameters,
      option->cse_prevention_only);

  if (pipeline.Run(module).status() != OkStatus()) {
    std::cerr << "Failed to run XLA sharding propagation pass." << std::endl;
    return XlaStatus::ERROR;
  }
  return XlaStatus::OK;
}

void xlaMakeDefaultSpmdPartitionerOption(XlaSpmdPartitionerOption* option) {
  option->num_partitions = -1;
  option->num_replicas = -1;
}

void xlaMakeDefaultCollectivesOptimizationPipeline(
    XlaCollectivesOptimizationOption* option) {
  option->reassociate_converted_all_reduce = true;
  option->enable_while_loop_reduce_scatter_code_motion = false;
  option->enable_data_parallel_collective_optimizer = false;
}

XlaStatus xlaRunCollectivesOptimizationPipeline(
    xla::HloModule* module, const XlaCollectivesOptimizationOption* option) {
  AlgebraicSimplifierOptions layout_insensitive_algsimp_opts(
      {}, gpu::GpuConvRewriter::ConvIsLowerable);
  HloPassPipeline pipeline("collectives-optimization");

  pipeline.AddPass<AllReduceFolder>();
  pipeline.AddPass<gpu::ReduceScatterCreator>();
  pipeline.AddPass<AllReduceReassociate>(
      option->reassociate_converted_all_reduce);
  pipeline.AddPass<ReduceScatterReassociate>();
  pipeline.AddPass<WhileLoopAllReduceCodeMotion>(
      /*enable_reduce_scatter=*/option
          ->enable_while_loop_reduce_scatter_code_motion);
  if (option->enable_data_parallel_collective_optimizer) {
    {
      CollectivePipeliner::Config config{
          /*op=*/HloOpcode::kAllReduce,
          /*level_to_operate_on=*/0,
          /*max_pipelining_per_loop=*/INT64_MAX,
          /*last_run=*/true,
          /*process_different_sized_ops=*/true,
          /*pipelining_direction=*/
          CollectivePipeliner::PipeliningDirection::kForward,
          /*should_process=*/HloPredicateIsOp<HloOpcode::kAllReduce>};
      pipeline.AddPass<CollectivePipeliner>(config);
    }
    {
      CollectivePipeliner::Config config{
          /*op=*/HloOpcode::kAllGather,
          /*level_to_operate_on=*/0,
          /*max_pipelining_per_loop=*/INT64_MAX,
          /*last_run=*/true,
          /*process_different_sized_ops=*/true,
          /*pipelining_direction=*/
          CollectivePipeliner::PipeliningDirection::kBackward,
          /*should_process=*/HloPredicateIsOp<HloOpcode::kAllReduce>};
      pipeline.AddPass<CollectivePipeliner>(config);
    }
    {
      CollectivePipeliner::Config config{
          /*op=*/HloOpcode::kReduceScatter,
          /*level_to_operate_on=*/0,
          /*max_pipelining_per_loop=*/INT64_MAX,
          /*last_run=*/true,
          /*process_different_sized_ops=*/true,
          /*pipelining_direction=*/
          CollectivePipeliner::PipeliningDirection::kForward,
          /*should_process=*/HloPredicateIsOp<HloOpcode::kAllReduce>};
      pipeline.AddPass<CollectivePipeliner>(config);
    }
  }

  // Run algebraic simplifier to reshape(broadcast) into a broadcast when
  // the reshape is just adding a unit dimension. This will help with the
  // AllGatherBroadcastReorder pass.
  pipeline.AddPass<AlgebraicSimplifier>(layout_insensitive_algsimp_opts);

  pipeline.AddPass<AllGatherBroadcastReorder>();

  if (pipeline.Run(module).status() != OkStatus()) {
    std::cerr << "Failed to run XLA collectives optimization pipeline."
              << std::endl;
    return XlaStatus::ERROR;
  }
  return XlaStatus::OK;
}

XlaStatus xlaRunSpmdPartitionerPass(xla::HloModule* module,
                                    const XlaSpmdPartitionerOption* option) {
  HloPassPipeline pipeline("spmd-partitioner");
  int64_t numPartitions = option->num_partitions == -1
                              ? module->config().num_partitions()
                              : option->num_partitions;
  int64_t numReplicas = option->num_replicas == -1
                            ? module->config().replica_count()
                            : option->num_replicas;
  pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(numPartitions,
                                                     numReplicas);
  pipeline.AddPass<CollectivePermuteMotion>();

  if (pipeline.Run(module).status() != OkStatus()) {
    std::cerr << "Failed to run XLA Stateful RNG SPMD partitioner pass."
              << std::endl;
    return XlaStatus::ERROR;
  }

  module->config().set_num_partitions(numPartitions);
  module->config().set_replica_count(numReplicas);

  return XlaStatus::OK;
}

XlaStatus xlaRunShardingPropagationAndSpmdPartitionerPasses(
    xla::HloModule* module,
    const XlaShardingPropagationOption* sharding_propagation_option,
    const XlaSpmdPartitionerOption* spmd_partitioner_option) {
  HloPassPipeline pipeline("sharding-propagation-spmd-partitioner");

  absl::Span<const bool> allow_spmd_sharding_propagation_to_output(
      sharding_propagation_option->allow_spmd_sharding_propagation_to_output,
      sharding_propagation_option
          ->allow_spmd_sharding_propagation_to_output_size);
  absl::Span<const bool> allow_spmd_sharding_propagation_to_parameters(
      sharding_propagation_option
          ->allow_spmd_sharding_propagation_to_parameters,
      sharding_propagation_option
          ->allow_spmd_sharding_propagation_to_parameters_size);
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<ZeroSizedHloElimination>();
  pipeline.AddPass<ConditionalCanonicalizer>();
  pipeline.AddPass<ShardingPropagation>(
      sharding_propagation_option->is_spmd,
      sharding_propagation_option->propagate_metadata,
      allow_spmd_sharding_propagation_to_output,
      allow_spmd_sharding_propagation_to_parameters,
      sharding_propagation_option->cse_prevention_only);

  int64_t numPartitions = spmd_partitioner_option->num_partitions == -1
                              ? module->config().num_partitions()
                              : spmd_partitioner_option->num_partitions;
  int64_t numReplicas = spmd_partitioner_option->num_replicas == -1
                            ? module->config().replica_count()
                            : spmd_partitioner_option->num_replicas;
  pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(numPartitions,
                                                     numReplicas);
  pipeline.AddPass<CollectivePermuteMotion>();

  if (pipeline.Run(module).status() != OkStatus()) {
    std::cerr << "Failed to run XLA Stateful RNG SPMD partitioner pass."
              << std::endl;
    return XlaStatus::ERROR;
  }

  module->config().set_num_partitions(numPartitions);
  module->config().set_replica_count(numReplicas);

  return XlaStatus::OK;
}

void xlaMakeDefaultAutoShardingOption(XlaAutoShardingOption* option) {
  option->enable = false;
  option->preserve_shardings =
      XlaAutoShardingOptionPreserveShardingsType::kKeepInputOutputShardings;
  option->simplify_graph = true;
  option->memory_budget_per_device = -1;
  option->memory_budget_ratio = 1.1;
  option->force_all_gather_cost = false;
  option->all_gather_cost = 0;
  option->force_all_to_all_cost = false;
  option->all_to_all_cost = 0;
  option->force_batch_dim_to_mesh_dim = -1;
  option->allow_replicated_parameters = true;
  option->prefer_reduce_scatter = false;
  option->reduce_scatter_grad_acc_friendly = false;
  option->reduce_scatter_aggressive_partition = false;
  option->batch_matmul_always_split_batch = false;
  option->allow_recompute_heavy_op = false;
  option->allow_mixed_mesh_shape = false;
  option->grad_acc_num_micro_batches = 1;
  option->load_solution_vector = false;
  option->solve_nd_sharding_iteratively = true;
  option->force_simple_heuristic = "";
  option->force_strategy = false;
  option->force_strategy_size = 0;
  option->force_strategy_inst_indices = nullptr;
  option->force_strategy_stra_names = nullptr;
  option->device_mesh_shape_size = 0;
  option->device_mesh_shape = nullptr;
  option->device_mesh_ids = nullptr;
  option->device_mesh_alpha = nullptr;
  option->device_mesh_beta = nullptr;
  option->load_strategy = false;
  option->try_multiple_mesh_shapes = false;
  option->solver_timeout_in_seconds = 3600;
  option->strategy_vector = nullptr;
  option->strategy_vector_size = 0;
}

XlaStatus xlaRunAutoShardingPass(xla::HloModule* module,
                                 const XlaAutoShardingOption* option) {
  HloPassPipeline pipeline("auto-sharding");
  xla::AutoShardingOption opt = convertAutoShardingOption(*option);
  pipeline.AddPass<xla::AutoSharding>(opt);
  if (pipeline.Run(module).status() != OkStatus()) {
    std::cerr << "Failed to run XLA Stateful RNG SPMD partitioner pass."
              << std::endl;
    return XlaStatus::ERROR;
  }
  return XlaStatus::OK;
}

}  // extern "C"
