#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include "common/common.h"


#define DPU_BINARY "/home/pribeiro/upmem_scripts/MLP/Blocking_DPU/mlp_dpu"
#define DPU_BINARY2 "/home/pribeiro/upmem_scripts/MLP/Blocking_DPU/mlp_dpu_other_func"


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
	DPU_ASSERT(dpu_alloc(2, NULL, set));
	DPU_ASSERT(dpu_load(*set, DPU_BINARY, NULL));

	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
}


/*float send_one_matrix()
{

}

float send_two_matrix()
{

}

float send_two_matrix_and transpose()
{

}*/

int main()
{
	srand(1);
	const int m_size = 60*50;
	const int m_size2 = 60*50;
	const int m_size3 = 60*50;
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



	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    alloc_dpus(&set, &nr_dpus);

    printf("DPUs allocated: %u\n", nr_dpus);

    //Load the DPUs with the binary
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[each_dpu* ((m_size)/nr_dpus)]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "d_test1", 0, sizeof(float) * ((m_size)/nr_dpus), DPU_XFER_DEFAULT));

    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &test2[each_dpu* ((m_size2)/nr_dpus)]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "d_test2", 0, sizeof(float) * ((m_size2)/nr_dpus), DPU_XFER_DEFAULT));

    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &res[each_dpu* ((m_size3)/nr_dpus)]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "d_res", 0, sizeof(float) * ((m_size3)/nr_dpus), DPU_XFER_DEFAULT));



	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));


	//Copy outputs from DPUS
	DPU_FOREACH(set,dpu, each_dpu)
	{
		DPU_ASSERT(dpu_copy_from(dpu, "d_res", 0, &res[each_dpu* ((m_size3)/nr_dpus)], sizeof(float) * ((m_size3)/nr_dpus)));
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}

	for(int i = 0; i < 60*50; i++)
	{
		printf("Result is : %.10f\n", res[i]);
	}


	//Load the DPUs with the binary
    DPU_ASSERT(dpu_load(set, DPU_BINARY2, NULL));

    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[each_dpu* ((m_size)/nr_dpus)]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "d_test1", 0, sizeof(float) * ((m_size)/nr_dpus), DPU_XFER_DEFAULT));

    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &test2[each_dpu* ((m_size2)/nr_dpus)]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "d_test2", 0, sizeof(float) * ((m_size2)/nr_dpus), DPU_XFER_DEFAULT));

    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &res[each_dpu* ((m_size3)/nr_dpus)]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "d_res", 0, sizeof(float) * ((m_size3)/nr_dpus), DPU_XFER_DEFAULT));

	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));


	//Copy outputs from DPUS
	DPU_FOREACH(set,dpu, each_dpu)
	{
		DPU_ASSERT(dpu_copy_from(dpu, "d_res", 0, &res[each_dpu* ((m_size3)/nr_dpus)], sizeof(float) * ((m_size3)/nr_dpus)));
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}

	for(int i = 0; i < 60*50; i++)
	{
		printf("Result is : %.10f\n", res[i]);
	}


	return 0;
}