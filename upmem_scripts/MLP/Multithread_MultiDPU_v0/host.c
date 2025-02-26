#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include <math.h>
#include "common/common.h"


#define KDOT_SIGMOID_BINARY "/home/pribeiro/upmem_scripts/MLP/Multithread_MultiDPU_v0/KDOT_SIGMOID"
#define KDOT_SIGMOID2_BINARY "/home/pribeiro/upmem_scripts/MLP/Multithread_MultiDPU_v0/KDOT_SIGMOID2"
#define KSUBTRACTION_KDOTELEMWISE_BINARY "../DPU_CODE/blablah.c"
#define KDOTM1M2T_KDOTELEMWISE_BINARY "../DPU_CODE/blablah.c"
#define KDOTM1TM2_BINARY "../DPU_CODE/blablah.c"

int Ceil(float a)
{
    return(a > (int) a) ? (a+1) : (a);
}


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
	DPU_ASSERT(dpu_load(*set, KDOT_SIGMOID_BINARY, NULL));
	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
}

void Run_KDot_Sigmoid(T *m1_, T *m2_, T *m3_, int m1_rows, int m1_cols, int m2_rows, int m2_cols, struct dpu_set_t set, struct dpu_set_t dpu, uint32_t nr_dpus, uint32_t each_dpu, char m1_name[], char m2_name[], char m3_name[], char binary_path[])
{
    
    int block1_rows = Ceil((T)m1_rows/NR_DPUS);
    int padding1 = block1_rows*NR_DPUS-m1_rows;
	int block2_cols = Ceil(((T)(m2_cols/NR_DPUS)/2))*2;
    int padding2 = block2_cols*NR_DPUS-m2_cols;
    printf("padding2: %d\n", padding2);
    T *m2 = (T*)malloc(sizeof(T)*(m2_rows)*(m2_cols+padding2));
    T *m1 = (T*)malloc(sizeof(T)*(m1_rows+padding1)*(m1_cols));
    T *m3 = (T*)malloc(sizeof(T)*(m1_rows+padding1)*(m2_cols+padding2));


    for(int i = 0; i < m1_rows*m1_cols; i++)
    {
    	m1[i] = m1_[i];
    }

    for(int i = 0; i < padding1*m1_cols; i++)
    {
    	m1[(m1_rows*m1_cols)+i] = 0.0;
    }

    int add_pad = 0;
    m2[0] = m2_[0];
   	for(int i = 1; i <= m2_rows*(m2_cols); i++)
    {
    	if((i%(m2_cols)) == 0)
    	{
    		//printf("padding2: %d", padding2);
    		for(int j = 1; j <= padding2; j++)
    		{
    			m2[i+add_pad] = 0.0;
    			add_pad++;
    				
    		}
    	}
    	m2[i+add_pad] = m2_[i];
    	//printf("%lf\n",test2[i+add_pad]);
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
    

    for(int i = 0; i < m2_rows; i++)
    {	
	    DPU_FOREACH(set,dpu,each_dpu)
    	{
    		DPU_ASSERT(dpu_prepare_xfer(dpu, &m2[(each_dpu%NR_DPUS)*block2_cols + (i*(m2_cols+padding2))]));
    	}
    	//DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[NR_DPUS* (last_block_rows*test1_cols)]));
    	//printf("line jump is: %d\n",i);
    	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m2_name, i*(block2_cols)*sizeof(T), sizeof(T) * ((block2_cols)), DPU_XFER_DEFAULT));
	}


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
    		
    		DPU_ASSERT(dpu_prepare_xfer(dpu, &m3[current_line*(m2_cols+padding2)+((each_dpu%NR_DPUS)*block2_cols)+((each_dpu/NR_DPUS)*block1_rows*(m2_cols+padding2))]));
    		//printf("val is: %d \n", current_line*(test2_cols+padding2)+((each_dpu%NR_DPUS)*block2_cols)+((each_dpu/NR_DPUS)*block1_rows*(test2_cols+padding2)));
    	//DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
    	}
    	//printf("for ended with line: %d\n", current_line);	
    	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, m3_name, current_line*(block2_cols)*sizeof(T), sizeof(T) * (block2_cols), DPU_XFER_DEFAULT));
    }

    /*for(int i = 0; i < m1_rows; i++)
    {
    	for(int j = 0; j < m2_cols; j++)
    	{
    		printf("%f ",m3[i*m2_cols+j]);
    	}
    	printf("\n");
    }*/

    //remove padding and copy to m3_
    //printf("\n\n\n\n");
    for(int i = 0; i < m1_rows; i++)
    {
    	for(int j = 0; j < m2_cols; j++)
    	{
    		m3_[i*m2_cols+j] = m3[i*m2_cols+(j+i*padding2)];
    		//printf("%f ", m3_[i*m2_cols+j]);
    	}
    	//printf("\n");
    }
}



int main()
{
	srand(1);
	
	T h_X_aux[TRAINING_SIZE*TRAINING_DIM] =
	{
5.8, 4.0, 1.2, 0.2,
5.7, 4.4, 1.5, 0.4,
5.4, 3.9, 1.3, 0.4,
5.1, 3.5, 1.4, 0.3,
5.7, 3.8, 1.7, 0.3,
5.1, 3.8, 1.5, 0.3,
5.4, 3.4, 1.7, 0.2,
5.1, 3.7, 1.5, 0.4,
4.6, 3.6, 1.0, 0.2,
5.1, 3.3, 1.7, 0.5,
4.8, 3.4, 1.9, 0.2,
5.0, 3.0, 1.6, 0.2,
5.0, 3.4, 1.6, 0.4,
5.2, 3.5, 1.5, 0.2,
5.2, 3.4, 1.4, 0.2,
4.7, 3.2, 1.6, 0.2,
4.8, 3.1, 1.6, 0.2,
5.4, 3.4, 1.5, 0.4,
5.2, 4.1, 1.5, 0.1,
5.5, 4.2, 1.4, 0.2,
4.9, 3.1, 1.5, 0.1,
5.0, 3.2, 1.2, 0.2,
5.5, 3.5, 1.3, 0.2,
4.9, 3.1, 1.5, 0.1,
4.4, 3.0, 1.3, 0.2,
5.1, 3.4, 1.5, 0.2,
5.0, 3.5, 1.3, 0.3,
4.5, 2.3, 1.3, 0.3,
4.4, 3.2, 1.3, 0.2,
5.0, 3.5, 1.6, 0.6,
5.1, 3.8, 1.9, 0.4,
4.8, 3.0, 1.4, 0.3,
5.1, 3.8, 1.6, 0.2,
4.6, 3.2, 1.4, 0.2,
5.3, 3.7, 1.5, 0.2,
5.0, 3.3, 1.4, 0.2,
7.0, 3.2, 4.7, 1.4,
6.4, 3.2, 4.5, 1.5,
6.9, 3.1, 4.9, 1.5,
5.5, 2.3, 4.0, 1.3,
6.5, 2.8, 4.6, 1.5,
5.7, 2.8, 4.5, 1.3,
6.3, 3.3, 4.7, 1.6,
4.9, 2.4, 3.3, 1.0,
6.6, 2.9, 4.6, 1.3,
5.2, 2.7, 3.9, 1.4,
5.0, 2.0, 3.5, 1.0,
5.9, 3.0, 4.2, 1.5,
6.0, 2.2, 4.0, 1.0,
6.1, 2.9, 4.7, 1.4,
5.6, 2.9, 3.6, 1.3,
6.7, 3.1, 4.4, 1.4,
5.6, 3.0, 4.5, 1.5,
5.8, 2.7, 4.1, 1.0,
6.2, 2.2, 4.5, 1.5,
5.6, 2.5, 3.9, 1.1,
5.9, 3.2, 4.8, 1.8,
6.1, 2.8, 4.0, 1.3,
6.3, 2.5, 4.9, 1.5,
6.1, 2.8, 4.7, 1.2,
6.4, 2.9, 4.3, 1.3,
6.6, 3.0, 4.4, 1.4,
6.8, 2.8, 4.8, 1.4,
6.7, 3.0, 5.0, 1.7,
6.0, 2.9, 4.5, 1.5,
5.7, 2.6, 3.5, 1.0,
5.5, 2.4, 3.8, 1.1,
5.5, 2.4, 3.7, 1.0,
5.8, 2.7, 3.9, 1.2,
6.0, 2.7, 5.1, 1.6,
5.4, 3.0, 4.5, 1.5,
6.0, 3.4, 4.5, 1.6,
6.7, 3.1, 4.7, 1.5,
6.3, 2.3, 4.4, 1.3,
5.6, 3.0, 4.1, 1.3,
5.5, 2.5, 4.0, 1.3,
5.5, 2.6, 4.4, 1.2,
6.1, 3.0, 4.6, 1.4,
5.8, 2.6, 4.0, 1.2,
5.0, 2.3, 3.3, 1.0,
5.6, 2.7, 4.2, 1.3,
5.7, 3.0, 4.2, 1.2,
5.7, 2.9, 4.2, 1.3,
6.2, 2.9, 4.3, 1.3,
5.1, 2.5, 3.0, 1.1,
5.7, 2.8, 4.1, 1.3,
6.3, 3.3, 6.0, 2.5,
5.8, 2.7, 5.1, 1.9,
7.1, 3.0, 5.9, 2.1,
6.3, 2.9, 5.6, 1.8,
6.5, 3.0, 5.8, 2.2,
7.6, 3.0, 6.6, 2.1,
4.9, 2.5, 4.5, 1.7,
7.3, 2.9, 6.3, 1.8,
6.7, 2.5, 5.8, 1.8,
7.2, 3.6, 6.1, 2.5,
6.5, 3.2, 5.1, 2.0,
6.4, 2.7, 5.3, 1.9,
6.8, 3.0, 5.5, 2.1,
5.7, 2.5, 5.0, 2.0,
5.8, 2.8, 5.1, 2.4,
6.4, 3.2, 5.3, 2.3,
6.5, 3.0, 5.5, 1.8,
7.7, 3.8, 6.7, 2.2,
7.7, 2.6, 6.9, 2.3,
6.0, 2.2, 5.0, 1.5,
6.9, 3.2, 5.7, 2.3,
5.6, 2.8, 4.9, 2.0,
7.7, 2.8, 6.7, 2.0,
6.3, 2.7, 4.9, 1.8,
6.7, 3.3, 5.7, 2.1,
7.2, 3.2, 6.0, 1.8,
6.2, 2.8, 4.8, 1.8,
6.1, 3.0, 4.9, 1.8,
6.4, 2.8, 5.6, 2.1,
7.2, 3.0, 5.8, 1.6,
7.4, 2.8, 6.1, 1.9,
7.9, 3.8, 6.4, 2.0,
6.4, 2.8, 5.6, 2.2,
6.3, 2.8, 5.1, 1.5,
6.1, 2.6, 5.6, 1.4,
7.7, 3.0, 6.1, 2.3};

	T* h_X = (T*)malloc(TRAINING_SIZE*TRAINING_DIM*sizeof(T));
	memcpy(h_X, h_X_aux, TRAINING_SIZE*TRAINING_DIM*sizeof(T));



	const signed int X_size = sizeof(h_X);
	T *d_X;

	//WEIGHTS_0
	const long signed int W0_size = L1_SIZE*TRAINING_DIM*sizeof(T);
	T *h_W0 = (T*)malloc(W0_size);
	for (int i = 0; i < L1_SIZE*TRAINING_DIM; i++){
	    h_W0[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	    //printf("val %d is: %.10f\n ", i, h_W0[i]);
	}



	//LAYER_1, LAYER_1_DELTA AND BUFFER OF LAYER 1 SIZE
	const long signed int L1_size = L1_SIZE*TRAINING_SIZE*sizeof(T);

	T* h_layer_1 = (T*)malloc(L1_size);
	T* h_layer_1_delta = (T*)malloc(L1_size);
	T* h_buffer = (T*)malloc(L1_size);

	for (int i = 0; i < L1_SIZE*TRAINING_SIZE; i++){
	    h_layer_1[i] = 0.0;
	    h_buffer[i] = 0.0;
	    h_layer_1_delta[i] = 0.0;
	}

	//WEIGHTS_1
	const long signed int W1_size = L1_SIZE*sizeof(T);
	T *h_W1 = (T*)malloc(W1_size);
	for (int i = 0; i < L1_SIZE; i++){
	    h_W1[i] = 0.1* (2.0*rand()/RAND_MAX-1.0);
	    printf("%f\n", h_W1[i]);
	}

	//Y
	//const int y_dim = 122;
	/*float h_Y[y_dim] = {0.0,
		0.0,
		1.0,
		1.0};*/
	T h_Y[y_dim] = {
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0};
	const signed int y_size = sizeof(h_Y);

	//PRED AND PRED_DELTA
	T* h_pred = (T*)malloc(y_size);
	T* h_pred_delta = (T*)malloc(y_size);
	for (int i = 0; i < TRAINING_SIZE; i++){
	    h_pred[i] = 0.0;
	    h_pred_delta[i] = 0.0;
	}

	//Allocate set of DPUs
	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    alloc_dpus(&set, &nr_dpus, NR_DPUS*NR_DPUS);

    printf("DPUs allocated: %u\n", nr_dpus);

    
    //Copy data
    //Note we can remove the copies with zeros
    /*DPU_ASSERT(dpu_copy_to(set, "d_X", 0, h_X, sizeof(float) * TRAINING_SIZE*TRAINING_DIM));
    DPU_ASSERT(dpu_copy_to(set, "d_W0", 0, h_W0, W0_size));

    DPU_ASSERT(dpu_copy_to(set, "d_layer_1", 0, h_layer_1, L1_size));
    DPU_ASSERT(dpu_copy_to(set, "d_buffer", 0, h_buffer, L1_size));
    DPU_ASSERT(dpu_copy_to(set, "d_layer_1_delta", 0, h_layer_1_delta, L1_size));

    DPU_ASSERT(dpu_copy_to(set, "d_W1", 0, h_W1, W1_size));

    DPU_ASSERT(dpu_copy_to(set, "d_Y", 0, h_Y, y_size));

    DPU_ASSERT(dpu_copy_to(set, "d_pred", 0, h_pred, y_size));
	DPU_ASSERT(dpu_copy_to(set, "d_pred_delta", 0, h_pred_delta, y_size));*/


	//test vector for KTest function
	const int test_size = 4*28;
	T h_test[test_size] = {5.1, 3.5, 1.4, 0.2,
4.9, 3.0, 1.4, 0.2,
4.7, 3.2, 1.3, 0.2,
4.6, 3.1, 1.5, 0.2,
5.0, 3.6, 1.4, 0.2,
5.4, 3.9, 1.7, 0.4,
4.6, 3.4, 1.4, 0.3,
5.0, 3.4, 1.5, 0.2,
4.4, 2.9, 1.4, 0.2,
4.9, 3.1, 1.5, 0.1,
5.4, 3.7, 1.5, 0.2,
4.8, 3.4, 1.6, 0.2,
4.8, 3.0, 1.4, 0.1,
4.3, 3.0, 1.1, 0.1,
6.3, 3.4, 5.6, 2.4,
6.4, 3.1, 5.5, 1.8,
6.0, 3.0, 4.8, 1.8,
6.9, 3.1, 5.4, 2.1,
6.7, 3.1, 5.6, 2.4,
6.9, 3.1, 5.1, 2.3,
5.8, 2.7, 5.1, 1.9,
6.8, 3.2, 5.9, 2.3,
6.7, 3.3, 5.7, 2.5,
6.7, 3.0, 5.2, 2.3,
6.3, 2.5, 5.0, 1.9,
6.5, 3.0, 5.2, 2.0,
6.2, 3.4, 5.4, 2.3,
5.9, 3.0, 5.1, 1.8};
	const int test_pred_size = 28;
	T h_test_pred[test_pred_size] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
	T h_test_l1[28*L1_SIZE];
	for (int i = 0; i < 28*L1_SIZE; i++){
	    h_test_l1[i] = 0;
	}

	//Copy test stuff
	//DPU_ASSERT(dpu_copy_to(set, "d_test", 0, h_test, sizeof(float)*test_size));
	//DPU_ASSERT(dpu_copy_to(set, "d_test_pred", 0, h_test_pred, sizeof(float)*test_pred_size));
	//DPU_ASSERT(dpu_copy_to(set, "d_test_l1", 0, h_test_l1, sizeof(float)*28*L1_SIZE));


	Run_KDot_Sigmoid(h_X, h_W0, h_layer_1, TRAINING_SIZE, TRAINING_DIM, TRAINING_DIM, L1_SIZE, set, dpu, nr_dpus, each_dpu, "d_X", "d_W0", "d_l1", KDOT_SIGMOID_BINARY);
	/*printf("\n\n\n\n");
	for(int i = 0; i < TRAINING_SIZE; i++)
	{
		for(int j = 0; j < L1_SIZE; j++)
		{
			printf("%f ", h_layer_1[i*L1_SIZE+j]);
		}
		printf("\n");
	}
	printf("\n\n\n\n");*/
	Run_KDot_Sigmoid(h_layer_1, h_W1, h_pred, TRAINING_SIZE, L1_SIZE, L1_SIZE, 1, set, dpu, nr_dpus, each_dpu, "d_l1", "d_W1", "d_pred", KDOT_SIGMOID2_BINARY);


	return 0;
}


