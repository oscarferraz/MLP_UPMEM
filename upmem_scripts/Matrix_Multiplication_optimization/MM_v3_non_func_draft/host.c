#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include <math.h>
#include "common/common.h"
#include "common/timer.h"

#define KDOT_BINARY "KDOT"
int Ceil(float a)
{
    return(a > (int) a) ? (a+1) : (a);
}


static void alloc_dpus(struct dpu_set_t *set, uint32_t *nr_dpus, int dpu_to_allocate)
{
	DPU_ASSERT(dpu_alloc(dpu_to_allocate, NULL, set));
	//DPU_ASSERT(dpu_load(*set, KDOT_BINARY, NULL));
	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
}

void Run_KDot(T *m1_, T *m2_, T *m3_, int m1_rows, int m1_cols, int m2_rows, int m2_cols, struct dpu_set_t set, struct dpu_set_t dpu, uint32_t nr_dpus, uint32_t each_dpu, char m1_name[], char m2_name[], char m3_name[], char binary_path[])
{

    //Note the blkocking must obey the following rules:
    //For m1:
    	//nr columns must be even
		//nr should be even but I don't think it is a must
	//For m2:
		//one of the dimensions must be the same as for m1 for multiplication to be possible.
		//the dimension that must be same depends if you want to transpose or not
	//The padding for matrix 2 since it will be the columns in the result matrix must occour in a way that it makes the number of
	//elements in the column even, that is why block2 is calculated diferently from block 1
    int block1_rows = Ceil((T)m1_rows/NR_DPUS);
    int padding1 = block1_rows*NR_DPUS-m1_rows;
	int block2_rows = Ceil((((T)m2_rows/NR_DPUS)/2))*2;
    int padding2 = block2_rows*NR_DPUS-m2_rows;
    //printf("padding2: %d\n", padding2);
    T *m1 = (T*)malloc(sizeof(T)*(m1_rows+padding1)*(m1_cols));
    T *m2 = (T*)malloc(sizeof(T)*(m2_rows+padding2)*(m2_cols));
    T *m3 = (T*)malloc(sizeof(T)*(m1_rows+padding1)*(m2_rows+padding2));


    for(int i = 0; i < m1_rows*m1_cols; i++)
    {
    	m1[i] = m1_[i];
    }

    for(int i = 0; i < padding1*m1_cols; i++)
    {
    	m1[(m1_rows*m1_cols)+i] = 0.0;
    }

    //matrix2 padding
    for(int i = 0; i < m2_rows*m2_cols; i++)
    {
    	m2[i] = m2_[i];
    }

    for(int i = 0; i < padding2*m2_cols; i++)
    {
    	m2[(m2_rows*m2_cols)+i] = 0.0;
    }



    //printf("You entered: %s\n.", binary_path);
    DPU_ASSERT(dpu_load(set, binary_path, NULL));
    //DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));

    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &m1[each_dpu/NR_DPUS* (block1_rows*m1_cols)]));
    }
    //DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[NR_DPUS* (last_block_rows*test1_cols)]));
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m1_name, 0, sizeof(T) * ((block1_rows*m1_cols)), DPU_XFER_DEFAULT));

	DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &m2[(each_dpu%NR_DPUS)*(block2_rows*m2_cols)]));
    }
    //DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[NR_DPUS* (last_block_rows*test1_cols)]));
    //printf("line jump is: %d\n",i);
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m2_name, 0, sizeof(T) * ((block2_rows*m2_cols)), DPU_XFER_DEFAULT));


	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

	//Print outputs from DPUS
	/*DPU_FOREACH(set,dpu,each_dpu)
    {
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}*/

    for(int current_line = 0; current_line < block1_rows; current_line++)
    {
    	DPU_FOREACH(set,dpu,each_dpu)
    	{
    		
    		DPU_ASSERT(dpu_prepare_xfer(dpu, &m3[current_line*(m2_rows+padding2)+((each_dpu%NR_DPUS)*block2_rows)+((each_dpu/NR_DPUS)*block1_rows*(m2_rows+padding2))]));
    		//printf("val is: %d \n", current_line*(test2_cols+padding2)+((each_dpu%NR_DPUS)*block2_cols)+((each_dpu/NR_DPUS)*block1_rows*(test2_cols+padding2)));
    	//DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
    	}
    	//printf("for ended with line: %d\n", current_line);	
    	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, m3_name, current_line*(block2_rows)*sizeof(T), sizeof(T) * (block2_rows), DPU_XFER_DEFAULT));
    }

    /*for(int i = 0; i < m1_rows+padding1; i++)
    {
    	for(int j = 0; j < m2_rows+padding2; j++)
    	{
    		printf("%.1f ",m3[i*(m2_rows+padding2)+j]);
    	}
    	printf("\n");
    }*/

    //remove padding and copy to m3_
    //printf("\n\n\n\n");
    for(int i = 0; i < m1_rows; i++)
    {
    	for(int j = 0; j < m2_rows; j++)
    	{
    		m3_[i*m2_rows+j] = m3[i*m2_rows+(j+i*padding2)];
    		//printf("%f ", m3_[i*m2_cols+j]);
    	}
    	//printf("\n");
    }
}

int main()
{
	
	T* m1 = (T*)malloc(M1_ROWS*M1_COLS*sizeof(T));
	for(int i = 0; i < M1_ROWS*M1_COLS; i++)
	{
		m1[i] = ((float)i);
	}
	
	//WEIGHTS_0
	const long signed int W0_size = M2_ROWS*M1_COLS*sizeof(T);
	T *m2 = (T*)malloc(W0_size);
	for(int i = 0; i < M2_ROWS*M1_COLS; i++)
	{
		m2[i] = ((float)i);
	}

	//LAYER_1, LAYER_1_DELTA AND BUFFER OF LAYER 1 SIZE
	const long signed int res_size = M1_ROWS*M2_ROWS*sizeof(T);

	//l1 variable is the output of the first layer after first sigmoid
	T* res = (T*)malloc(res_size);



 	Timer timer;
	start(&timer, 0, 0);

	//Allocate set of DPUs
	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    alloc_dpus(&set, &nr_dpus, NR_DPUS*NR_DPUS);


	Run_KDot(m1, m2, res, M1_ROWS, M1_COLS, M2_ROWS, M1_COLS, set, dpu, nr_dpus, each_dpu, "d_m1", "d_m2", "d_res", KDOT_BINARY);
	printf("\n\n\n\n");
	for(int i = 0; i < M1_ROWS; i++)
	{
		for(int j = 0; j < M2_ROWS; j++)
		{
			printf("%.1f ", res[i*M2_ROWS+j]);
		}
		printf("\n");
	}
	printf("\n\n\n\n");
  	stop(&timer, 0);
	printf("GEMM Time (ms): ");
	print(&timer, 0, 1);

	/*printf("\n\nPrediction values are: \n");
	for(int i = 0; i < TRAINING_SIZE; i++)
	{
		printf("%d: %f\n",i, h_pred[i]);
	}
	printf("\n\n");*/
	return 0;
}


