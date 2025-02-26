#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include "common/common.h"
//#include "math.h"


#include <stdbool.h>
#include <string.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <math.h>


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

int main()
{

	const long signed int test_size1 = test1_rows*test1_cols*sizeof(T);
	const long signed int test_size2 = test2_rows*test2_cols*sizeof(T);

	T *test1_ = (T*)malloc(test_size1);
	T *test2_ = (T*)malloc(test_size2);
	for (int i = 0; i < test2_rows*test2_cols; i++)
	{
	    test2_[i] = i+1;
	    //printf("%.f\n", test2_[i]);
	}

	for (int i = 0; i < test1_rows*test1_cols; i++)
	{
	    test1_[i] = i+1;
	    //printf("%.f\n", test2_[i]);
	}

	//Allocate set of DPUs
	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    alloc_dpus(&set, &nr_dpus, NR_DPUS*NR_DPUS);
 
    printf("DPUs allocated: %u\n", nr_dpus);
    
    //for double
    //int block2_cols = ceil((float)test2_cols/NR_DPUS);
    //int padding2 = test2_cols%NR_DPUS;

    //for Float
    int block2_cols = ceil(((float)(test2_cols/NR_DPUS)/2))*2;
    int padding2 = block2_cols*NR_DPUS-test2_cols;

    //int block1_rows = ceil(((float)(test1_rows/NR_DPUS)/2))*2;
    //int padding1 = block1_rows*NR_DPUS-test1_rows;
    int block1_rows = ceil((float)test1_rows/NR_DPUS);
    int padding1 = block1_rows*NR_DPUS-test1_rows;

    T *test2 = (T*)malloc(sizeof(T)*(test2_rows)*(test2_cols+padding2));
    T *test1 = (T*)malloc(sizeof(T)*(test1_rows+padding1)*(test1_cols));
    T *res = (T*)malloc(sizeof(T)*(test1_rows+padding1)*(test2_cols+padding2));

    for(int i = 0; i < test1_rows*test1_cols; i++)
    {
    	test1[i] = test1_[i];
    }


    for(int i = 0; i < padding1*test1_cols; i++)
    {
    	test1[(test1_rows*test1_cols)+i] = 0.0;
    }

    int add_pad = 0;
    test2[0] = test2_[0];
    printf("\n\n\n\n\n\n\ntest2[0] printed: %f\n\n\n\n\n\n\n", test2[0]);
	for(int i = 1; i <= test2_rows*(test2_cols); i++)
    {
    	if((i%(test2_cols)) == 0)
    	{
    		//printf("padding2: %d", padding2);
    		for(int j = 1; j <= padding2; j++)
    		{
    			test2[i+add_pad] = 0.0;
    			add_pad++;
    				
    		}
    	}
    	test2[i+add_pad] = test2_[i];
    	//printf("%lf\n",test2[i+add_pad]);
    }

    /*for(int i = 0; i < (test2_rows); i++)
    {
    	for(int j = 0; j < (test2_cols+padding2); j++)
    	{
    		printf("Value for vector %d is: %lf\n",i,test2[i*(test2_cols+padding2)+j]);
    	}
    }*/
    //printf("padding2 value is %d\n", block2_cols);

    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[each_dpu/NR_DPUS* (block1_rows*test1_cols)]));
    }
    //DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[NR_DPUS* (last_block_rows*test1_cols)]));
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "d_test1", 0, sizeof(T) * ((block1_rows*test1_cols)), DPU_XFER_DEFAULT));
    
    for(int i = 0; i < test2_rows; i++)
    {	
	    DPU_FOREACH(set,dpu,each_dpu)
    	{
    		DPU_ASSERT(dpu_prepare_xfer(dpu, &test2[(each_dpu%NR_DPUS)*block2_cols + (i*(test2_cols+padding2))]));
    	}
    	//DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[NR_DPUS* (last_block_rows*test1_cols)]));
    	//printf("line jump is: %d\n",i);
    	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "d_test2", i*(block2_cols)*sizeof(T), sizeof(T) * ((block2_cols)), DPU_XFER_DEFAULT));
	}

    /*DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &res[each_dpu* (((test1_rows+padding1)*(test2_cols+padding2))/nr_dpus)]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "d_res", 0, sizeof(float) * (((test1_rows+padding1)*(test2_cols+padding2))/nr_dpus), DPU_XFER_DEFAULT));
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
    		
    		DPU_ASSERT(dpu_prepare_xfer(dpu, &res[current_line*(test2_cols+padding2)+((each_dpu%NR_DPUS)*block2_cols)+((each_dpu/NR_DPUS)*block1_rows*(test2_cols+padding2))]));
    		//printf("val is: %d \n", current_line*(test2_cols+padding2)+((each_dpu%NR_DPUS)*block2_cols)+((each_dpu/NR_DPUS)*block1_rows*(test2_cols+padding2)));
    	//DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
    	}
    	//printf("for ended with line: %d\n", current_line);	
    	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "d_res", current_line*(block2_cols)*sizeof(T), sizeof(T) * (block2_cols), DPU_XFER_DEFAULT));
    }

    //printf("\n\n\n%d\n\n\n", padding1);

	/*for(int i = 0; i < test1_rows+padding1; i++)
	{
		//printf("%d ",i);
		for(int j = 0; j < test2_cols+padding2; j++)
		{
			printf("%.1f ", res[i*(test2_cols+padding2)+j]);
		}
		printf("\n");
	}*/

	return 0;
}
