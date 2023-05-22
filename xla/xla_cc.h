#ifndef XLA_XLA_CC_H
#define XLA_XLA_CC_H

#include <stdint.h>
#include <stdlib.h>

namespace xla {
class HloModule;
}  // namespace xla

extern "C" {

enum XlaStatus { OK = 0, ERROR = 1 };

enum XlaAutoShardingOptionPreserveShardingsType {
  // AutoSharding constrains the search space using all user shardings.
  kKeepAllShardings,
  // AutoSharding constains the search space using input and output shardings
  // of HloModule's entry computations and remove shardings of all
  // intermediate tensors.
  kKeepInputOutputShardings,
  // Remove all user shardings. This is useful when testing with HLO
  // modules with XLA shardings, so that we can get performance comparison
  // with
  // and without AutoSharding, without changing HLO Modules.
  kRemoveAllShardings
};

typedef struct XlaAutoShardingOption {
  // Enable the auto sharding pass.
  bool enable = false;

  XlaAutoShardingOptionPreserveShardingsType preserve_shardings;

  // Simplify the cost graph by merging nodes that should have the same sharding
  // strategy. E.g., an XLAop constructed from an elementwise transformation of
  // another XLAop.
  bool simplify_graph;

  // Memory budget (bytes) per device. Default value -1 means no memory budget.
  // Value 0 means setting it to the memory lower bound estimation.
  int64_t memory_budget_per_device;

  // Memory budget =
  //     memory_budget_ratio * (memory lower bound estimation).
  // Enabled when memory_budget_per_device == 0;
  float memory_budget_ratio;

  // Overwrite the all gather cost with the input all reduce cost.
  bool force_all_gather_cost;
  double all_gather_cost;

  // Overwrite the all gather cost with the input all reduce cost.
  bool force_all_to_all_cost;
  double all_to_all_cost;

  // Forcibly split the batch dimension and map it to a mesh dimension.
  // This can force the auto-sharding pass to generate the data parallel
  // strategy.
  int force_batch_dim_to_mesh_dim;

  // If true, allow replicated parameters.
  bool allow_replicated_parameters;

  // If true, prefer reduce-scatter + all-gather over all-reduce.
  // A post process will be applied to replace all-reduce with reduce-scater +
  // all-gather if no communication overhead is introduced.
  bool prefer_reduce_scatter;

  // If True, generate a gradient-accumulation friendly variant of
  // reduce-scatter
  bool reduce_scatter_grad_acc_friendly;

  // If true, aggressively partition more tensors when generating
  // reduce-scatter, even if it introduces more communication.
  bool reduce_scatter_aggressive_partition;

  // If true, the batch matmul will always be parallelized on the batch dim in
  // 2d mesh case.
  bool batch_matmul_always_split_batch;

  // If true, allow strategies that recompute heavy operators (e.g., dot)
  // to reduce communication.
  bool allow_recompute_heavy_op;

  // If true, allow adding 1d strategies in 2d logical mesh.
  bool allow_mixed_mesh_shape;

  // The number of micro batches if gradient accumulation is used.
  // If this is not 1, the cost of all-reduce for gradient synchronization
  // is divided by this number.
  int grad_acc_num_micro_batches;

  // If true, load solution vector from PassContext
  bool load_solution_vector;

  // If true, N-D sharding (e.g., N maybe be 2 or 3) will be solved in N
  // iterations, where one iteration chooses one tensor dimension to shard. If
  // false, solve N-D sharding directly, i.e., generating all possible sharding
  // strategies for N-D mesh shape.
  bool solve_nd_sharding_iteratively;

  // If it is not empty, forcibly use simple heuristic strategies
  // instead of the ILP solver. This is used for ablation study.
  char* force_simple_heuristic;

  // If true, forcibly set the strategy of some instructions.
  bool force_strategy;
  size_t force_strategy_size;
  int64_t* force_strategy_inst_indices;
  char** force_strategy_stra_names;

  // Device mesh shape.
  int64_t* device_mesh_shape;
  size_t device_mesh_shape_size;
  // Device IDs in the mesh.
  int64_t* device_mesh_ids;
  // We use an alpha-beta model as the communication model:
  //   latency = alpha + beta * size
  // the following two vectors have the same size as device_mesh_shape and each
  // element models the communication performance along each mesh dimension.
  double* device_mesh_alpha;
  double* device_mesh_beta;
  // Load the strategy vector instead of solving one.
  bool load_strategy = false;
  // Explore other mesh shapes with the same number of devices as the provided
  // one for a potentially better auto-sharding solution.
  bool try_multiple_mesh_shapes = false;

  // Timeout for the solver. If the solver fails to find an optimal solution
  // before the timeout, we rely on the heuristic-based sharding implemented in
  // sharding_propagation.cc.
  int64_t solver_timeout_in_seconds = 3600;
  int64_t* strategy_vector;
  size_t strategy_vector_size;
} XlaAutoShardingOption;

typedef struct XlaShardingPropagationOption {
  bool is_spmd;
  bool propagate_metadata;
  bool* allow_spmd_sharding_propagation_to_output;
  int64_t allow_spmd_sharding_propagation_to_output_size;
  bool* allow_spmd_sharding_propagation_to_parameters;
  int64_t allow_spmd_sharding_propagation_to_parameters_size;
  bool cse_prevention_only;
} XlaShardingPropagationOption;

typedef struct XlaSpmdPartitionerOption {
  int64_t num_partitions;
  int64_t num_replicas;
} XlaSpmdPartitionerOption;

XlaStatus xlaStableHloFileToXlaHlo(const char* filepath,
                                   xla::HloModule** outModule);
XlaStatus xlaStableHloBufferToXlaHlo(const char* mlirBytecodeBuffer,
                                     size_t mlirBytecodeBufferSize,
                                     xla::HloModule** outModule);
void xlaDestroyHloModule(xla::HloModule* hloModule);
void xlaDestroyCharBuffer(char* buff);
XlaStatus xlaXlaHloToStableHloBuffer(const xla::HloModule& module,
                                     char** outMlirBytecodeBuffer,
                                     size_t* outMlirBytecodeBufferSize);

void xlaMakeDefaultShardingPropagationOption(
    XlaShardingPropagationOption* option);
XlaStatus xlaRunShardingPropagationPass(
    xla::HloModule* module, const XlaShardingPropagationOption* option);

void xlaMakeDefaultSpmdPartitionerOption(XlaSpmdPartitionerOption* option);
XlaStatus xlaRunSpmdPartitionerPass(xla::HloModule* module,
                                    const XlaSpmdPartitionerOption* option);

void xlaMakeDefaultAutoShardingOption(XlaAutoShardingOption* option);
XlaStatus xlaRunAutoShardingPass(xla::HloModule* module,
                                 const XlaAutoShardingOption* option);
}

#endif  // XLA_XLA_CC_H
