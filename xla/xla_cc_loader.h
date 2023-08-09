#ifndef XLA_XLA_CC_LOADER_H
#define XLA_XLA_CC_LOADER_H

#include <dlfcn.h>
#include <stdlib.h>

#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "xla_cc.h"

namespace xla {
namespace api {

using StableHloFileToXlaHlo = XlaStatus (*)(const char*, xla::HloModule**);
using StableHloBufferToXlaHlo = XlaStatus (*)(const char*, size_t,
                                              xla::HloModule**);
using DestroyHloModule = void (*)(HloModule*);
using DestroyCharBuffer = void (*)(char*);
using XlaHloToStableHloBuffer = XlaStatus (*)(const xla::HloModule&, char**,
                                              size_t*);
using MakeDefaultShardingPropagationOption =
    void (*)(XlaShardingPropagationOption* option);
using RunShardingPropagationPass = XlaStatus (*)(
    xla::HloModule* module, const XlaShardingPropagationOption* option);
using MakeDefaultSpmdPartitionerOption =
    void (*)(XlaSpmdPartitionerOption* option);
using RunSpmdPartitionerPass = XlaStatus (*)(
    xla::HloModule* module, const XlaSpmdPartitionerOption* option);
using RunShardingPropagationAndSpmdPartitionerPasses = XlaStatus (*)(
    xla::HloModule* module,
    const XlaShardingPropagationOption* sharding_propagation_option,
    const XlaSpmdPartitionerOption* spmd_partitioner_option);
using MakeDefaultCollectivesOptimizationPipeline =
    void (*)(XlaCollectivesOptimizationOption* option);
using RunCollectivesOptimizationPipeline = XlaStatus (*)(
    xla::HloModule* module, const XlaCollectivesOptimizationOption* option);
using MakeDefaultAutoShardingOption = void (*)(XlaAutoShardingOption* option);
using RunAutoShardingPass = XlaStatus (*)(xla::HloModule* module,
                                          const XlaAutoShardingOption* option);

using ParseHloSharding = XlaStatus (*)(const char* str, size_t strSize,
                                       xla::HloSharding** outSharding);
using DestroyHloSharding = void (*)(xla::HloSharding* sharding);
using HloShardingIsTuple = bool (*)(const xla::HloSharding* sharding);
using HloShardingIsTiled = bool (*)(const xla::HloSharding* sharding);
using HloShardingIsReplicated = bool (*)(const xla::HloSharding* sharding);
using HloShardingIsManual = bool (*)(const xla::HloSharding* sharding);
using HloShardingReplicateOnLastTileDim =
    bool (*)(const xla::HloSharding* sharding);
using HloShardingTileAssignmentDevices =
    void (*)(const xla::HloSharding* sharding, const int64_t** outDevices,
             size_t* outDevicesSize);
using HloShardingTileAssignmentDevicesShape =
    void (*)(const xla::HloSharding* sharding, const int64_t** outShape,
             size_t* outShapeSize);
using HloShardingTileShape = void (*)(const xla::HloSharding* sharding,
                                      const int64_t* tensorShape,
                                      size_t tensorShapeSize,
                                      int64_t* outTileShape,
                                      size_t* outTileShapeSize);

extern StableHloFileToXlaHlo stableHloFileToXlaHlo;
extern StableHloBufferToXlaHlo stableHloBufferToXlaHlo;
extern DestroyHloModule destroyHloModule;
extern DestroyCharBuffer destroyCharBuffer;
extern XlaHloToStableHloBuffer xlaHloToStableHloBuffer;
extern MakeDefaultShardingPropagationOption
    makeDefaultShardingPropagationOption;
extern RunShardingPropagationPass runShardingPropagationPass;
extern MakeDefaultSpmdPartitionerOption makeDefaultSpmdPartitionerOption;
extern RunSpmdPartitionerPass runSpmdPartitionerPass;
extern RunShardingPropagationAndSpmdPartitionerPasses
    runShardingPropagationAndSpmdPartitionerPasses;
extern MakeDefaultCollectivesOptimizationPipeline
    makeDefaultCollectivesOptimizationPipeline;
extern RunCollectivesOptimizationPipeline runCollectivesOptimizationPipeline;
extern MakeDefaultAutoShardingOption makeDefaultAutoShardingOption;
extern RunAutoShardingPass runAutoShardingPass;

extern ParseHloSharding parseHloSharding;
extern DestroyHloSharding destroyHloSharding;
extern HloShardingIsTuple hloShardingIsTuple;
extern HloShardingIsTiled hloShardingIsTiled;
extern HloShardingIsReplicated hloShardingIsReplicated;
extern HloShardingIsManual hloShardingIsManual;
extern HloShardingReplicateOnLastTileDim hloShardingReplicateOnLastTileDim;
extern HloShardingTileAssignmentDevices hloShardingTileAssignmentDevices;
extern HloShardingTileAssignmentDevicesShape
    hloShardingTileAssignmentDevicesShape;
extern HloShardingTileShape hloShardingTileShape;

XlaStatus loadSymbol(void* libraryHandle, void*& dst, const char* symbol);
XlaStatus loadSymbols(void* libraryHandle);
std::shared_ptr<void> loadLibrary(const char* path);

}  // namespace api
}  // namespace xla

#endif  // XLA_XLA_CC_LOADER_H
