#ifndef THRUSTSORT_H
#define THRUSTSORT_H

#include <cstring>
#include <cub/util_type.cuh>
#include "tiny_obj_loader.h"

cudaError_t cuda_kernel(tinyobj::attrib_t attrib, std::vector<tinyobj::shape_t> shapes);

#endif
/*
template <typename K>
void RandomBits(
		K &key,
		int entropy_reduction = 0,
		int begin_bit = 0,
		int end_bit = sizeof(K) * 8)
{
  const int NUM_BYTES = sizeof(K);
  const int WORD_BYTES = sizeof(unsigned int);
  const int NUM_WORDS = (NUM_BYTES + WORD_BYTES - 1) / WORD_BYTES;

  unsigned int word_buff[NUM_WORDS];

  if (entropy_reduction == -1)
    {
      memset((void *) &key, 0, sizeof(key));
      return;
    }

  if (end_bit < 0)
    end_bit = sizeof(K) * 8;

  while (true)
    {
      // Generate random word_buff
      for (int j = 0; j < NUM_WORDS; j++)
        {
	  int current_bit = j * WORD_BYTES * 8;

	  unsigned int word = 0xffffffff;
	  word &= 0xffffffff << CUB_MAX(0, begin_bit - current_bit);
	  word &= 0xffffffff >> CUB_MAX(0, (current_bit + (WORD_BYTES * 8)) - end_bit);

	  for (int i = 0; i <= entropy_reduction; i++)
            {
	      // Grab some of the higher bits from rand (better entropy, supposedly)
	      word &= mersenne::genrand_int32();
	      g_num_rand_samples++;
            }

	  word_buff[j] = word;
        }

      memcpy(&key, word_buff, sizeof(K));

      K copy = key;
      if (!IsNaN(copy))
	break;          // avoids NaNs when generating random floating point numbers
    }
}
*/
