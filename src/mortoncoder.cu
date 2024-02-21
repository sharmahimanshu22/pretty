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
#include <cub/cub.cuh> 

using namespace cub;
//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

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

__device__ 
void findSplit(uint64_t* codes, int nodeId, int direction, int l, int& childLeft, int& childRight) {

  // l can be two or greater
  int t = (l+1)/2;
  int s = 0;
  uint64_t code1 = codes[nodeId];
  uint64_t code2 = codes[nodeId + (l-1)*direction];
  int minCodeLength = __clzll(code1^code2);
  while(true) {
    uint64_t other = codes[nodeId + (s+t)*direction];
    int commonCodeLength = __clzll(code^other);
    if (commonCodeLength > minCodeLength) {
      s += t;
      if (t == 1) {
	break;
      }
      t = (t+1)/2;
    }
  }
}


__device__
void findRangeWithSamePrefix(const uint64_t* codes, const int count, int nodeId,  unsigned int* lengths) {
  

  // get direction and mincommonlength
  
  assert (nodeId <= count - 2);
  assert (nodeId >= 1);
  uint64_t code = codes[nodeId];
  uint64_t codeleft = codes[nodeId-1];
  uint64_t coderight = codes[nodeId+1];
  int commonLengthLeft = __clzll(code^codeleft);
  int commonLengthRight = __clzll(code^coderight);
  if(nodeId == 8084) {
    printf("%d \n", commonLengthLeft);
    printf("%d \n\n\n", commonLengthRight);
  }

  int direction = commonLengthRight > commonLengthLeft ? 1 : -1; // The should not be equal. Equal means there is a code repetition.
  int commonLengthMinBound = direction == 1 ? commonLengthLeft : commonLengthRight;
  int commonLength = direction == 1 ? commonLengthRight : commonLengthLeft;


  // find lmax
  int lmax = 2;
  while(true) {
    int idx = nodeId + lmax*direction;
    if (idx > count - 1 || idx < 0) {
      break;
    }
    uint64_t code2 = codes[idx];
    commonLength = __clzll(code^code2);
    if (commonLength > commonLengthMinBound) {
      lmax += lmax;
    } else {
      break;
    }
  }
  // We have lmax. Real l value is between lmax/2 and lmax;

  // find l
  int l = 0;
  int t = lmax/2;
  while(t >= 1) {
    int idx = nodeId + (l+t)*direction;
    if (idx < 0 || idx > count-1) {
      t = t/2;
    } else {
      uint64_t code2 = codes[idx];
      commonLength = __clzll(code^code2);
      if(commonLength > commonLengthMinBound) {
	l = l + t;
      }
      t = t/2;
    }
  }
  l = l+1; // This is just the adjustment for acual length. The previous while loop computes the length of range - 1.

  lengths[nodeId] = l;
  // Now we have the final value of l
  
}



// https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf
__global__
void createBVH(uint64_t* sortedMortonCodes, int count, unsigned int* lengths) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int nt = blockDim.x*gridDim.x;
  
  for (int i = tid; i < count-2; i = i + nt) {
    findRangeWithSamePrefix(sortedMortonCodes, count, i+1, lengths);  // i == 0 will be dealt separately
  }
  lengths[0] = count;

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
void cuda_memsetindices(int* idces, int n) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int nt = blockDim.x*gridDim.x;

  for(int i = tid; i < n; i = i+nt) { 
    idces[i] = i;
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

/*
The method updates the parameters mortonCondes and sortedOrder
 */
__host__
void radixSortMortonCodes(uint64_t* mortonCodes, int* &sortedOrder, const int numOfPrimitives) {

  cudaError_t error = cudaSuccess;
  int* idces;
  error = cudaMalloc(&idces, numOfPrimitives*sizeof(int));
  cuda_memsetindices<<<32,64>>>(idces, numOfPrimitives);

  int* idcesAlternate;
  error = cudaMalloc(&idcesAlternate, numOfPrimitives*sizeof(int));
  cudaMemcpy(idcesAlternate, idces, numOfPrimitives*sizeof(int), cudaMemcpyDeviceToDevice);

  uint64_t* mortonCodesAlternate;
  error = cudaMalloc(&mortonCodesAlternate, numOfPrimitives*sizeof(uint64_t));
  cudaMemcpy(mortonCodesAlternate, mortonCodes, numOfPrimitives*sizeof(uint64_t), cudaMemcpyDeviceToDevice);

  cub::DoubleBuffer<uint64_t> d_keys = *(new cub::DoubleBuffer<uint64_t>(mortonCodes, mortonCodesAlternate));
  cub::DoubleBuffer<int> d_values= *(new cub::DoubleBuffer<int>(idces, idcesAlternate));

  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, numOfPrimitives);
  
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, numOfPrimitives);

  if(error) {
    std::cout << cudaGetErrorString(error) << " memset failed\n";
    return;
  }

  sortedOrder = d_values.Current();
  mortonCodes = d_keys.Current();

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
  
  int* sortedIndices = NULL;
  radixSortMortonCodes(mortonCodes, sortedIndices, nFaces);

  error = cudaGetLastError();
  if(error) {
    std::cout << cudaGetErrorString(error) << " sorting failed\n";
    return error;
  }

  float* centroids = (float*)malloc(3*nFaces*sizeof(float));
  cudaMemcpy(centroids, d_centroids, 3*nFaces*sizeof(float), cudaMemcpyDeviceToHost);
  uint64_t* mortonhost = (uint64_t*)malloc(nFaces*sizeof(uint64_t));
  cudaMemcpy(mortonhost, mortonCodes, nFaces*sizeof(uint64_t), cudaMemcpyDeviceToHost);

  int* sortedHost = (int*) malloc(nFaces*sizeof(int));
  cudaMemcpy(sortedHost, sortedIndices, nFaces*sizeof(int), cudaMemcpyDeviceToHost);


  std::cout << std::bitset<64> (17) << " :check this\n"  ;
  for(int kk = 0; kk < 20; kk++) {
    std::cout << std::bitset<64>(mortonhost[kk]);
    std:: cout << "," << mortonhost[kk] << "\n" ;  //<< std::bitset<64>(mortonhost[idx]) << "\n";
  }

  cudaDeviceSynchronize();


  unsigned int* lengths;
  cudaMalloc(&lengths, (nFaces-1)*sizeof(unsigned int));

  createBVH<<<32,64>>>(mortonCodes, nFaces, lengths);
  
  int* lengthsHost = (int*) malloc((nFaces-1)*sizeof(unsigned int));
  cudaMemcpy(lengthsHost, lengths, (nFaces-1)*sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "\n\n\n";
  for(int kk = 8080; kk < 8090; kk++) {
    //if (lengthsHost[kk] > 1000) {
    std::cout << kk << " " << lengthsHost[kk] << " " << mortonhost[kk] << " " << mortonhost[kk-1] << " " << mortonhost[kk+1] << "\n";      
      //}
    //std:: cout << "," << sortedHost[kk] << "\n" ;  //<< std::bitset<64>(mortonhost[idx]) << "\n";
  }



  return error;
}
