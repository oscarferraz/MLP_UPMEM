#include <stdio.h>
#include <stdlib.h>
#include "common/common.h"
#include "upmem_kernels2.c"
#include <time.h>

#define NR_DPUS 256

#define rows 2048
#define cols 2048

#define m_size rows * cols
#define m_size2 rows * cols
#define m_size3 rows * cols

static void free_buffers(uint32_t *input_array_1, uint32_t *input_array_2, uint32_t *output_array)
{
	free(input_array_1);
	free(input_array_2);
	free(output_array);
}


int main()
{
  const long signed int test_size = m_size*sizeof(float);
	const long signed int test_size2 = m_size2*sizeof(float);
	const long signed int test_size3 = m_size3*sizeof(float);

	float *test1 = (float*)malloc(test_size);
	for (int i = 0; i < m_size; i++)
	{
	    //test1[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	    test1[i] = i;
	    //printf("%.10f ", test1[i]);
	}

	float *test2 = (float*)malloc(test_size2);
	for (int i = 0; i < m_size2; i++)
	{
	    //test2[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	    test2[i] = 5.0;
	    //printf("%.10f ", h_W0[i]);
	}

	float *res = (float*)malloc(test_size3);
	for (int i = 0; i < m_size3; i++)
	{
	    res[i] = 0.0;
	    //printf("%.10f ", h_W0[i]);
	}

	 clock_t start, end;
     double cpu_time_used;
     
     start = clock();

     kMartixByMatrixElementwise(rows, cols, test1, test2, res);

	end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("MLP %f seconds to execute \n", cpu_time_used);

	return 0;
}
