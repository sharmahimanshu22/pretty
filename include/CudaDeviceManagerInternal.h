// -*- c++ -*-
// Ensure printing of CUDA runtime errors to console
#ifndef CUDADEVICEMANAGER_H
#define CUDADEVICEMANAGER_H

#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>

class CudaDeviceManagerInternal{
  inline static cudaDeviceProp              deviceProp;
  inline static float                       device_giga_bandwidth;
  inline static std::size_t                 device_free_physmem;
  inline static std::size_t                 device_total_physmem;

public:
  static cudaError_t initDevice(int dev);

};

#endif
