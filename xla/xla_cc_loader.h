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
using MakeDefaultCollectivesOptimizationPipeline =
    void (*)(XlaCollectivesOptimizationOption* option);
using RunCollectivesOptimizationPipeline = XlaStatus (*)(
    xla::HloModule* module, const XlaCollectivesOptimizationOption* option);
using MakeDefaultAutoShardingOption = void (*)(XlaAutoShardingOption* option);
using RunAutoShardingPass = XlaStatus (*)(xla::HloModule* module,
                                          const XlaAutoShardingOption* option);

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
extern MakeDefaultCollectivesOptimizationPipeline
    makeDefaultCollectivesOptimizationPipeline;
extern RunCollectivesOptimizationPipeline runCollectivesOptimizationPipeline;
extern MakeDefaultAutoShardingOption makeDefaultAutoShardingOption;
extern RunAutoShardingPass runAutoShardingPass;

XlaStatus loadSymbol(void* libraryHandle, void*& dst, const char* symbol);
XlaStatus loadSymbols(void* libraryHandle);
std::shared_ptr<void> loadLibrary(const char* path);

}  // namespace api
}  // namespace xla

#endif  // XLA_XLA_CC_LOADER_H
