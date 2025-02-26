#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include "common/common.h"
#include "math.h"

static void free_buffers(uint32_t *input_array_1, uint32_t *input_array_2, uint32_t *output_array)
{
	free(input_array_1);
	free(input_array_2);
	free(output_array);
}



static void free_dpus(struct dpu_set_t set)
{}

static void alloc_dpus(struct dpu_set_t *set, uint32_t *nr_dpus, int dpu_to_allocate)
{
	DPU_ASSERT(dpu_alloc(dpu_to_allocate, NULL, set));
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
	const int m_size = test1_rows*test1_cols;
	const int m_size2 = test2_rows*test2_cols;
	const int m_size3 = test1_rows*test2_cols;
	const long signed int test_size = m_size*sizeof(float);
	const long signed int test_size2 = m_size2*sizeof(float);
	const long signed int test_size3 = m_size3*sizeof(float);

	float *test1_ = (float*)malloc(test_size);
	for (int i = 0; i < m_size; i++)
	{
	    //test1[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	    test1_[i] = i;
	    //printf("%.10f ", test1[i]);
	}

	float *test2_ = (float*)malloc(test_size2);
	for (int i = 0; i < m_size2; i++)
	{
	    //test2[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	    test2_[i] = 5.0;
	    //printf("%.10f ", h_W0[i]);
	}

	float *res = (float*)malloc(test_size3);
	for (int i = 0; i < m_size3; i++)
	{
	    res[i] = 0.0;
	    //printf("%.10f ", h_W0[i]);
	}

	//Allocate set of DPUs
	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    alloc_dpus(&set, &nr_dpus, NR_DPUS*NR_DPUS);
 
    printf("DPUs allocated: %u\n", nr_dpus);
    
    int block1_rows = ceil((float)test1_rows/NR_DPUS);
    int padding1 = test1_rows%NR_DPUS;
    int block2_rows = ceil((float)test2_rows/NR_DPUS);
    int padding2 = test2_rows%NR_DPUS;
	//Note: we are considering the second matrix already transposed
    float *test1 = (float*)realloc(test1_, sizeof(float)*(test1_rows+padding1)*test1_cols);
    float *test2 = (float*)realloc(test2_, sizeof(float)*(test2_rows+padding2)*test2_cols);

    for(int i = 0; i < padding1*test1_cols; i++)
    {
    	test1[m_size+i] = 0.0;
    }
    for(int i = 0; i < padding2*test2_cols; i++)
    {
    	test2[m_size2+i] = 0.0;
    }

    /*for(int i = 0; i < (test2_rows+padding2)*test2_cols; i++)
    {
    	printf("Value for vector 2 is: %f\n",test2[i]);
    }*/

    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[each_dpu/NR_DPUS* (block1_rows*test1_cols)]));
    }
    //DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[NR_DPUS* (last_block_rows*test1_cols)]));
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "d_test1", 0, sizeof(float) * ((block1_rows*test1_cols)), DPU_XFER_DEFAULT));

    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &test2[each_dpu%NR_DPUS* (block2_rows*test2_cols)]));
    }
    //DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[NR_DPUS* (last_block_rows*test1_cols)]));
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "d_test2", 0, sizeof(float) * ((block2_rows*test2_cols)), DPU_XFER_DEFAULT));
	

    /*DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &res[each_dpu* ((m_size3)/nr_dpus)]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "d_res", 0, sizeof(float) * ((m_size3)/nr_dpus), DPU_XFER_DEFAULT));
	*/


	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));


	//Copy outputs from DPUS
	DPU_FOREACH(set,dpu,each_dpu)
    {
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}

    for(int current_line = 0; current_line < block1_rows; current_line++)
    {
    	DPU_FOREACH(set,dpu,each_dpu)
    	{
    	
    		DPU_ASSERT(dpu_prepare_xfer(dpu, &res[current_line*(test2_rows+padding2)+((each_dpu%NR_DPUS)*block2_rows)+((each_dpu/NR_DPUS)*block1_rows*(test2_rows+padding2))]));
    	
    	//DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
    	}	
    	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "d_res", current_line*(test2_rows+padding2), sizeof(float) * (block2_rows), DPU_XFER_DEFAULT));
    }


	for(int i = 0; i < 70+padding1; i++)
	{
		printf("%d ",i);
		for(int j = 0; j < 14+padding2; j++)
		{
			printf("%.1f ", res[i*(14+padding2)+j]);
		}
		printf("\n");
	}

	return 0;
}