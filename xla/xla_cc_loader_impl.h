#include <dlfcn.h>
#include <stdlib.h>

#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "xla_cc_loader.h"

namespace xla {
namespace api {

StableHloFileToXlaHlo stableHloFileToXlaHlo = nullptr;
StableHloBufferToXlaHlo stableHloBufferToXlaHlo = nullptr;
DestroyHloModule destroyHloModule = nullptr;
DestroyCharBuffer destroyCharBuffer = nullptr;
XlaHloToStableHloBuffer xlaHloToStableHloBuffer = nullptr;
MakeDefaultShardingPropagationOption makeDefaultShardingPropagationOption =
    nullptr;
RunShardingPropagationPass runShardingPropagationPass = nullptr;
MakeDefaultSpmdPartitionerOption makeDefaultSpmdPartitionerOption = nullptr;
RunSpmdPartitionerPass runSpmdPartitionerPass = nullptr;
MakeDefaultCollectivesOptimizationPipeline
    makeDefaultCollectivesOptimizationPipeline = nullptr;
RunCollectivesOptimizationPipeline runCollectivesOptimizationPipeline = nullptr;
MakeDefaultAutoShardingOption makeDefaultAutoShardingOption = nullptr;
RunAutoShardingPass runAutoShardingPass = nullptr;

XlaStatus loadSymbol(void* libraryHandle, void*& dst, const char* symbol) {
  dst = dlsym(libraryHandle, symbol);
  if (!dst) {
    std::cerr << "Failed loading symbol \"" << symbol << "\": " << dlerror()
              << std::endl;
    return XlaStatus::ERROR;
  }
  return XlaStatus::OK;
}

XlaStatus loadSymbols(void* libraryHandle) {
  std::vector<std::tuple<const char*, void**>> symbolTable = {
      {"xlaStableHloFileToXlaHlo",
       reinterpret_cast<void**>(&stableHloFileToXlaHlo)},
      {"xlaStableHloBufferToXlaHlo",
       reinterpret_cast<void**>(&stableHloBufferToXlaHlo)},
      {"xlaDestroyHloModule", reinterpret_cast<void**>(&destroyHloModule)},
      {"xlaDestroyCharBuffer", reinterpret_cast<void**>(&destroyCharBuffer)},
      {"xlaXlaHloToStableHloBuffer",
       reinterpret_cast<void**>(&xlaHloToStableHloBuffer)},
      {"xlaMakeDefaultShardingPropagationOption",
       reinterpret_cast<void**>(&makeDefaultShardingPropagationOption)},
      {"xlaRunShardingPropagationPass",
       reinterpret_cast<void**>(&runShardingPropagationPass)},
      {"xlaMakeDefaultSpmdPartitionerOption",
       reinterpret_cast<void**>(&makeDefaultSpmdPartitionerOption)},
      {"xlaRunSpmdPartitionerPass",
       reinterpret_cast<void**>(&runSpmdPartitionerPass)},
      {"xlaMakeDefaultCollectivesOptimizationPipeline",
       reinterpret_cast<void**>(&makeDefaultCollectivesOptimizationPipeline)},
      {"xlaRunCollectivesOptimizationPipeline",
       reinterpret_cast<void**>(&runCollectivesOptimizationPipeline)},
      {"xlaMakeDefaultAutoShardingOption",
       reinterpret_cast<void**>(&makeDefaultAutoShardingOption)},
      {"xlaRunAutoShardingPass",
       reinterpret_cast<void**>(&runAutoShardingPass)}};
  for (auto& sym : symbolTable) {
    XlaStatus status =
        loadSymbol(libraryHandle, *std::get<1>(sym), std::get<0>(sym));
    if (status != XlaStatus::OK) {
      return status;
    }
  }
  return XlaStatus::OK;
}

std::shared_ptr<void> loadLibrary(const char* path) {
  static std::weak_ptr<void> weakHandle;
  std::shared_ptr<void> handle = weakHandle.lock();
  if (handle) {
    return handle;
  }

  handle = std::shared_ptr<void>(
      dlopen(path, RTLD_DEEPBIND | RTLD_LOCAL | RTLD_LAZY), [](void* h) {
        if (h) {
          dlclose(h);
        }
      });
  if (!handle) {
    std::cerr << "Failed loading library \"" << path << "\": " << dlerror()
              << std::endl;
    std::cerr << "Maybe you forgot to add a its path to the OS runtime library "
                 "search paths."
              << std::endl;
    std::cerr << "On Unix-like systems you can use the environment variable "
                 "LD_LIBRARY_PATH."
              << std::endl;
    return nullptr;
  }

  XlaStatus status = loadSymbols(handle.get());
  if (status != XlaStatus::OK) {
    std::cerr << "Failed loading library \"" << path << "\": " << dlerror()
              << std::endl;
    return nullptr;
  }
  weakHandle = handle;

  return handle;
}

}  // namespace api
}  // namespace xla
