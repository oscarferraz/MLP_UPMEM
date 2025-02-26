#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include "common/common.h"


static void free_buffers(uint32_t *input_array_1, uint32_t *input_array_2, uint32_t *output_array)
{
	free(input_array_1);
	free(input_array_2);
	free(output_array);
}



static void free_dpus(struct dpu_set_t set)
{}

static void alloc_dpus(struct dpu_set_t *set, uint32_t *nr_dpus)
{
	DPU_ASSERT(dpu_alloc(1, NULL, set));
	DPU_ASSERT(dpu_load(*set, DPU_BINARY, NULL));

	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
}


int main()
{
	srand(1);
	const int m_size1 = M1_ROWS*M1_COLS;
	const int m_size2 = M2_ROWS*M2_COLS;
	const int m_size3 = M1_ROWS*M2_COLS;
	const long signed int test_size1 = m_size1*sizeof(float);
	const long signed int test_size2 = m_size2*sizeof(float);
	const long signed int test_size3 = m_size3*sizeof(float);

	float *test1 = (float*)malloc(test_size1);
	for (int i = 0; i < m_size1; i++)
	{
	    //test1[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	    test1[i] = 2.0;
	    //printf("%.10f ", test1[i]);
	}

	float *test2 = (float*)malloc(test_size2);
	for (int i = 0; i < m_size2; i++)
	{
	    //test2[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	    test2[i] = 1.0;
	    //printf("%.10f ", h_W0[i]);
	}

	float *res = (float*)malloc(test_size3);
	for (int i = 0; i < m_size3; i++)
	{
	    res[i] = 0.0;
	    //printf("%.10f ", h_W0[i]);
	}



	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    alloc_dpus(&set, &nr_dpus);

    printf("DPUs allocated: %u\n", nr_dpus);

    
    DPU_ASSERT(dpu_copy_to(set, "d_test1", 0, test1, test_size1));
    DPU_ASSERT(dpu_copy_to(set, "d_test2", 0, test2, test_size2));
    DPU_ASSERT(dpu_copy_to(set, "d_res", 0, res, test_size3));

    //Run DPUS
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));


	//Copy outputs from DPUS
	DPU_FOREACH(set,dpu)
	{
		//DPU_ASSERT(dpu_copy_from(dpu, "X_res", 0, test_X, sizeof(float) * TRAINING_SIZE*TRAINING_DIM));
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}

	return 0;
}