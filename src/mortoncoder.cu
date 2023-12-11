// -*- c++ -*-

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#include <stdio.h>
#include <algorithm>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "util.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cmath>
#include "mortoncoder.h"
#include <bitset>

using namespace cub;
//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory
//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/*
struct Pair
{
    float   key;
    int     value;
    bool operator<(const Pair &b) const
    {
        if (key < b.key)
            return true;
        if (key > b.key)
            return false;
        // Return true if key is negative zero and b.key is positive zero
        unsigned int key_bits   = SafeBitCast<unsigned int>(key);
        unsigned int b_key_bits = SafeBitCast<unsigned int>(b.key);
        unsigned int HIGH_BIT   = 1u << 31;
        return ((key_bits & HIGH_BIT) != 0) && ((b_key_bits & HIGH_BIT) == 0);
    }
};



void Initialize(
    float           *h_keys,
    int             *h_values,
    float           *h_reference_keys,
    int             *h_reference_values,
    int             num_items)
{
    Pair *h_pairs = new Pair[num_items];
    for (int i = 0; i < num_items; ++i)
    {
        RandomBits(h_keys[i]);
        RandomBits(h_values[i]);
        h_pairs[i].key    = h_keys[i];
        h_pairs[i].value  = h_values[i];
    }
    if (g_verbose)
    {
        printf("Input keys:\n");
        DisplayResults(h_keys, num_items);
        printf("\n\n");
        printf("Input values:\n");
        DisplayResults(h_values, num_items);
        printf("\n\n");
    }
    std::stable_sort(h_pairs, h_pairs + num_items);
    for (int i = 0; i < num_items; ++i)
    {
        h_reference_keys[i]     = h_pairs[i].key;
        h_reference_values[i]   = h_pairs[i].value;
    }
    delete[] h_pairs;
}

*/
/*

https://stackoverflow.com/a/17401122
__device__ 
static float atomicMax(float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
		      __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ 
static float atomicMin(float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
		      __float_as_int(::fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}
*/

// Refer following for morton encoding.
//https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
// https://github.com/Forceflow/libmorton  - This is a good library for CPUs
__device__ 
void splitBy3(int a, uint64_t& s) {
    s = a & 0x1fffff;
    s = (s | s << 32) & 0x1f00000000ffff;
    s = (s | s << 16) & 0x1f0000ff0000ff;
    s = (s | s << 8) & 0x100f00f00f00f00f;
    s = (s | s << 4) & 0x10c30c30c30c30c3;
    s = (s | s << 2) & 0x1249249249249249;
}




__global__
void compute64BitMortonCode3dPoint(float* centroids, int count, int* bboxint, uint64_t* mortonCodes)
{
  //printf("here1\n");
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int nt = blockDim.x*gridDim.x;
  
  for (int i = tid; i < count; i = i + nt) {

    uint64_t s_x;
    uint64_t s_y;
    uint64_t s_z;

    int cx_int = __float2int_rn(max((centroids[3*i] - bboxint[0])*4194303/(bboxint[3] - bboxint[0]), 0.0));
    cx_int = cx_int <= 4194303 ? cx_int : 4194303;   // This is just to make sure there is no numerical error 
    int cy_int = __float2int_rn(max((centroids[3*i+1] - bboxint[1])*4194303/(bboxint[4] - bboxint[1]), 0.0));
    cy_int = cy_int <= 4194303 ? cy_int : 4194303;
    int cz_int = __float2int_rn(max((centroids[3*i+2] - bboxint[2])*4194303/(bboxint[5] - bboxint[2]), 0.0));
    cz_int = cz_int <= 4194303 ? cz_int : 4194303;
    
    splitBy3(cx_int, s_x);
    splitBy3(cy_int, s_y);
    splitBy3(cz_int, s_z);
    mortonCodes[i] = s_x | s_y << 1 | s_z << 2 ;
    
  }
  
  
}



__global__
void computeTriangleCentroids(float* d_vertices, int* d_indices, float* d_centroids, int nFaces, int* bbox) 
{
  
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int nt = blockDim.x*gridDim.x;
  
  for(int i = tid; i < nFaces; i = i+nt) { 
    int idx1 = d_indices[3*i + 0];
    int idx2 = d_indices[3*i + 1];
    int idx3 = d_indices[3*i + 2];

    float x1 = d_vertices[3*idx1 + 0];
    float y1 = d_vertices[3*idx1 + 1];
    float z1 = d_vertices[3*idx1 + 2];

    float x2 = d_vertices[3*idx2 + 0];
    float y2 = d_vertices[3*idx2 + 1];
    float z2 = d_vertices[3*idx2 + 2];

    float x3 = d_vertices[3*idx3 + 0];
    float y3 = d_vertices[3*idx3 + 1];
    float z3 = d_vertices[3*idx3 + 2];
    
    float c_x = (x1 + x2 + x3)/3.0;
    float c_y = (y1 + y2 + y3)/3.0;
    float c_z = (z1 + z2 + z3)/3.0;

    d_centroids[3*i+0] = c_x;
    d_centroids[3*i+1] = c_y;
    d_centroids[3*i+2] = c_z;

    
    atomicMin(&bbox[0], __float2int_rd(c_x));
    atomicMin(&bbox[1], __float2int_rd(c_y)); 
    atomicMin(&bbox[2], __float2int_rd(c_z)); 

    atomicMax(&bbox[3], __float2int_ru(c_x));
    atomicMax(&bbox[4], __float2int_ru(c_y)); 
    atomicMax(&bbox[5], __float2int_ru(c_z));
    
  }
 
}

#include <cassert>
#include <cmath>
//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
__host__
void copyTinyObjSceneToGPU(tinyobj::attrib_t& attrib, std::vector<tinyobj::shape_t>& shapes, float* &d_vertices, int* &d_indices, int &nFaces, cudaError_t &error) {
			   //std::vector<tinyobj::material_t>& meterials = NULL) {
  // For now maybe just copy vertices and face indices for only first shape ?
  // They will fire me for writing this code.

  error = cudaMalloc(&d_vertices, attrib.vertices.size()*sizeof(float));
  if(error) return;
  error = cudaMemcpy(d_vertices, attrib.vertices.data(), attrib.vertices.size()*sizeof(float), cudaMemcpyHostToDevice);
  if(error) return;

  int n = shapes[0].mesh.indices.size();
  assert (n%3 == 0);
  nFaces = n/3;
  std::vector<int> vertex_indices(n);
  
  std::transform(shapes[0].mesh.indices.cbegin(), shapes[0].mesh.indices.cend(), vertex_indices.cbegin(),
		 vertex_indices.begin(), [](tinyobj::index_t idx, int i) -> int { return idx.vertex_index; });

  error = cudaMalloc(&d_indices, n*sizeof(int));
  if(error) return;
  error = cudaMemcpy(d_indices, vertex_indices.data(), n*sizeof(int), cudaMemcpyHostToDevice);
  if(error) return;
  
  return;
}




cudaError_t cuda_kernel(tinyobj::attrib_t attrib, std::vector<tinyobj::shape_t> shapes)
{

  float* d_vertices;
  int* d_indices;
  int nFaces = -1;
  cudaError_t error = cudaSuccess;

  
  error = cudaMalloc(&d_vertices, attrib.vertices.size()*sizeof(float));
  if(error) return error;
  error = cudaMemcpy(d_vertices, attrib.vertices.data(), attrib.vertices.size()*sizeof(float), cudaMemcpyHostToDevice);
  if(error) return error;

  copyTinyObjSceneToGPU(attrib, shapes, d_vertices, d_indices, nFaces, error);
  if(error) return error;

  int blockSize;      // The launch configurator returned block size 
  int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
  int gridSize;       // The actual grid size needed, based on input size 

  error = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeTriangleCentroids, 0, nFaces); 
  if(error) return error;

  std::cout << "Suggested block Size: " << blockSize << "\n";
  std::cout << "Suggested mingrid Size: " << minGridSize << "\n";

  float* d_centroids;
  error = cudaMalloc(&d_centroids, 3*nFaces*sizeof(float));
  if(error) {
    std::cout << cudaGetErrorString(error);
    std::cout <<  "exit from here\n";
    return error;
  }

  int* bbox;
  error = cudaMalloc(&bbox, 6*sizeof(int));
  if(error) {
    std::cout <<  "exit from here22\n";
    return error;
  }
  int inftyup = std::numeric_limits<int>::max();
  int inftydown = std::numeric_limits<int>::min();
  int* bboxstart = new int[6];
  bboxstart[0] = inftyup;
  bboxstart[1] = inftyup;
  bboxstart[2] = inftyup;
  bboxstart[3] = inftydown;
  bboxstart[4] = inftydown;
  bboxstart[5] = inftydown;

  error = cudaMemcpy(bbox, bboxstart, 6*sizeof(int), cudaMemcpyHostToDevice);

  if(error) {
    std::cout <<  "exit from here2\n";
    return error;
  }
  dim3 dimGrid(minGridSize,1,1);
  dim3 dimBlock(blockSize,1,1);

  cudaDeviceSynchronize();

  error = cudaGetLastError();
  if(error) {
    std::cout << cudaGetErrorString(error) << " before first\n";
    return error;
  }
  std::cout << "centroid nfaces " << nFaces << "\n";
  computeTriangleCentroids<<<32, 64>>>(d_vertices, d_indices, d_centroids, nFaces, bbox);
  cudaDeviceSynchronize();

  error = cudaGetLastError();
  if(error) {
    std::cout << cudaGetErrorString(error) << " first\n";
    return error;
  }

  int* hostbbox = (int*)malloc(6*sizeof(int));
  cudaMemcpy(hostbbox, bbox, 6*sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "bbox : " << hostbbox[0] << ","<< hostbbox[1] << ","<< hostbbox[2] << ","<< hostbbox[3] << ","<< hostbbox[4] << ","<< hostbbox[5] << "\n";

  uint64_t* mortonCodes;
  error = cudaMalloc(&mortonCodes, nFaces*sizeof(uint64_t));
  if(error) return error;

  error = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, compute64BitMortonCode3dPoint, 0, nFaces); 
  if(error) return error;
  dim3 dimGrid2(32,1,1);
  dim3 dimBlock2(32,1,1);

  cudaDeviceSynchronize();
  compute64BitMortonCode3dPoint<<<dimGrid2, dimBlock2>>>(d_centroids, nFaces, bbox, mortonCodes);
  cudaDeviceSynchronize();

  error = cudaGetLastError();
  if(error) {
    std::cout << cudaGetErrorString(error) << " over here\n";
    return error;
  }
  
  
  float* centroids = (float*)malloc(3*nFaces*sizeof(float));
  cudaMemcpy(centroids, d_centroids, 3*nFaces*sizeof(float), cudaMemcpyDeviceToHost);
  uint64_t* mortonhost = (uint64_t*)malloc(nFaces*sizeof(uint64_t));
  cudaMemcpy(mortonhost, mortonCodes, nFaces*sizeof(uint64_t), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  for(int i = 0; i < 100; i++) {
    int idx1 = shapes[0].mesh.indices[i].vertex_index;
    int idx2 = shapes[0].mesh.indices[i+1].vertex_index;
    int idx3 = shapes[0].mesh.indices[i+2].vertex_index;
    
    float x1 = attrib.vertices[3*idx1 +0];
    float y1 = attrib.vertices[3*idx1 +1];
    float z1 = attrib.vertices[3*idx1 +2];

    float x2 = attrib.vertices[3*idx2 +0];
    float y2 = attrib.vertices[3*idx2 +1];
    float z2 = attrib.vertices[3*idx2 +2];

    float x3 = attrib.vertices[3*idx3 +0];
    float y3 = attrib.vertices[3*idx3 +1];
    float z3 = attrib.vertices[3*idx3 +2];


    std::cout << x1 << " " << x2 << " " << x3 << y1 << " " << y2 << " " << y3 << z1 << " " << z2 << " " << z3 << "\n";
    std::cout << centroids[3*i]  << " " << centroids[3*i + 1]  << " " << centroids[3*i+2] << " " << std::bitset<64>(mortonhost[i]) << "\n";
  }
  




  /*
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  int blockSize;      // The launch configurator returned block size 
  int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
  int gridSize;       // The actual grid size needed, based on input size 

  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeTriangleCentroids, 0, nFaces); 

  std::cout << "minGridSize: " << minGridSize << " blockSize: " << blockSize << "\n";
  */

  //computeTriangleCentroids(d_v, d_indices, centroid, )
  
  



  
  /*
    int num_items = 150;
    // Initialize command line
    printf("cub::DeviceRadixSort::SortPairs() %d items (%d-byte keys %d-byte values)\n",
        num_items, int(sizeof(float)), int(sizeof(int)));
    fflush(stdout);
    // Allocate host arrays
    float   *h_keys             = new float[num_items];
    float   *h_reference_keys   = new float[num_items];
    int     *h_values           = new int[num_items];
    int     *h_reference_values = new int[num_items];
    // Initialize problem and solution on host

    
    Initialize(h_keys, h_values, h_reference_keys, h_reference_values, num_items);


    
    // Allocate device arrays
    DoubleBuffer<float> d_keys;
    DoubleBuffer<int>   d_values;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(float) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(float) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(int) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(int) * num_items));
    // Allocate temporary storage
    size_t  temp_storage_bytes  = 0;
    void    *d_temp_storage     = NULL;
    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    // Initialize device arrays
    CubDebugExit(cudaMemcpy(d_keys.d_buffers[d_keys.selector], h_keys, sizeof(float) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values.d_buffers[d_values.selector], h_values, sizeof(int) * num_items, cudaMemcpyHostToDevice));
    // Run
    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));
    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_reference_keys, d_keys.Current(), num_items, true, g_verbose);
    printf("\t Compare keys (selector %d): %s\n", d_keys.selector, compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);
    compare = CompareDeviceResults(h_reference_values, d_values.Current(), num_items, true, g_verbose);
    printf("\t Compare values (selector %d): %s\n", d_values.selector, compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);
    // Cleanup
    if (h_keys) delete[] h_keys;
    if (h_reference_keys) delete[] h_reference_keys;
    if (h_values) delete[] h_values;
    if (h_reference_values) delete[] h_reference_values;
    if (d_keys.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[0]));
    if (d_keys.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
    if (d_values.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[0]));
    if (d_values.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    printf("\n\n");

    */
    return error;
}
