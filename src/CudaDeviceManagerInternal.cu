// -*- c++ -*-
// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include "CudaDeviceManagerInternal.h"
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>


cudaError_t CudaDeviceManagerInternal::initDevice(int dev) {

    cudaError_t error = cudaSuccess;

    int deviceCount;
    error = CubDebug(cudaGetDeviceCount(&deviceCount));
    if (error) return error;
    
    if (deviceCount == 0) {
      fprintf(stderr, "No devices supporting CUDA.\n");
      exit(1);
    }
    
    if (dev < 0 || dev > deviceCount - 1) {
      dev = 0;
    }
    
    error = CubDebug(cudaSetDevice(dev));
    if (error) return error;

    CubDebugExit(cudaMemGetInfo(&device_free_physmem, &device_total_physmem));

    int ptx_version = 0;
    error = CubDebug(CUB_NS_QUALIFIER::PtxVersion(ptx_version));
    if (error) return error;

    error = CubDebug(cudaGetDeviceProperties(&deviceProp, dev));
    if (error) return error;

    if (deviceProp.major < 1) {
      fprintf(stderr, "Device does not support CUDA.\n");
      exit(1);
    }

    device_giga_bandwidth = float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000;

    printf(
	   "Using device %d: %s (PTX version %d, SM%d, %d SMs, "
	   "%lld free / %lld total MB physmem, "
	   "%.3f GB/s @ %d kHz mem clock, ECC %s)\n",
	   dev,
	   deviceProp.name,
	   ptx_version,
	   deviceProp.major * 100 + deviceProp.minor * 10,
	   deviceProp.multiProcessorCount,
	   (unsigned long long) device_free_physmem / 1024 / 1024,
	   (unsigned long long) device_total_physmem / 1024 / 1024,
	   device_giga_bandwidth,
	   deviceProp.memoryClockRate,
	   (deviceProp.ECCEnabled) ? "on" : "off");
    fflush(stdout);

    return error;
  }
