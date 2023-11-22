// -*- c++ -*-
#include <iostream>
#include "commandlineargsparser.h"
#include <stdio.h>
#include <algorithm>
#include "CudaDeviceManagerInternal.h"
#include <cuda_runtime.h>
#include "mortoncoder.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

//bool                    g_verbose = false;  // Whether to display input/output to console
//CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Simple key-value pairing for floating point types.  Distinguishes
 * between positive and negative zero.
 */
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
*/


/**
 * Initialize key-value sorting problem.
 */

/*
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




//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char** argv)
{
  int num_items = 150;

  // Initialize command line
  CommandLineArgsParser parser = CommandLineArgsParser(argc, argv);

  // Initialize device
  cudaError_t error = CudaDeviceManagerInternal::initDevice(0);
  
  std::string inputfile = "neptune.obj";
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string warn;
  std::string err;

  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());

  if (!warn.empty()) {
    std::cout << warn << std::endl;
  }

  if (!err.empty()) {
    std::cerr << err << std::endl;
  }

  if (!ret) {
    exit(1);
  }


  cuda_kernel(attrib, shapes);
  
  // Loop over shapes
  for (size_t s = 0; s < shapes.size(); s++) {
    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

      // Loop over vertices in the face.
      for (size_t v = 0; v < fv; v++) {
	// access to vertex
	tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

	tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
	tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
	tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];

	// Check if `normal_index` is zero or positive. negative = no normal data
	if (idx.normal_index >= 0) {
	  tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
	  tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
	  tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];
	}

	// Check if `texcoord_index` is zero or positive. negative = no texcoord data
	if (idx.texcoord_index >= 0) {
	  tinyobj::real_t tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
	  tinyobj::real_t ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];
	}
	// Optional: vertex colors
	// tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
	// tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
	// tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
      }
      index_offset += fv;

      // per-face material
      shapes[s].mesh.material_ids[f];
    }
  }

  std::cout << "All went fine it seems\n";

  /*
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

  return 0;
}


int main2(int argc, char** argv) {

  CommandLineArgsParser parser = CommandLineArgsParser(argc, argv);
  parser.printAllArgs();
  std::string val = parser.getCommandLineArg<std::string>("hello");
  std::cout << "The key is hello. The value is " << val << "\n"; 
  return 0;
}
