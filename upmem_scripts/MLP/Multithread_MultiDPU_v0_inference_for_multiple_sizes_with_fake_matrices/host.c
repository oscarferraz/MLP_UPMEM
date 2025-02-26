#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include "common/common.h"
#include "common/timer.h"

#define KDOT_SIGMOID_BINARY "KDOT_SIGMOID"
#define KDOT_SIGMOID2_BINARY "KDOT_SIGMOID2"

int Ceil(float a)
{
    return(a > (int) a) ? (a+1) : (a);
}

static void alloc_dpus(struct dpu_set_t *set, uint32_t *nr_dpus, int dpu_to_allocate)
{
	DPU_ASSERT(dpu_alloc(dpu_to_allocate, NULL, set));
	//DPU_ASSERT(dpu_load(*set, KDOT_SIGMOID_BINARY, NULL));
	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
}

void Run_KDot_Sigmoid(T *m1_, T *m2_, T *m3_, int m1_rows, int m1_cols, int m2_rows, int m2_cols, struct dpu_set_t set, struct dpu_set_t dpu, uint32_t nr_dpus, uint32_t each_dpu, char m1_name[], char m2_name[], char m3_name[], char binary_path[])
{
    
    int block1_rows = Ceil((T)m1_rows/NR_DPUS);
    int padding1 = block1_rows*NR_DPUS-m1_rows;
	int block2_cols = Ceil((((T)m2_cols/NR_DPUS)/2))*2;
    int padding2 = block2_cols*NR_DPUS-m2_cols;
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
    		for(int j = 1; j <= padding2; j++)
    		{
    			m2[i+add_pad] = 0.0;
    			add_pad++;	
    		}
    	}
    	m2[i+add_pad] = m2_[i];
    }

    //load binary
    DPU_ASSERT(dpu_load(set, binary_path, NULL));


    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &m1[each_dpu/NR_DPUS* (block1_rows*m1_cols)]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m1_name, 0, sizeof(T) * ((block1_rows*m1_cols)), DPU_XFER_DEFAULT));
    
    for(int i = 0; i < m2_rows; i++)
    {	
	    DPU_FOREACH(set,dpu,each_dpu)
    	{
    		DPU_ASSERT(dpu_prepare_xfer(dpu, &m2[(each_dpu%NR_DPUS)*block2_cols + (i*(m2_cols+padding2))]));
    	}
    	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m2_name, i*(block2_cols)*sizeof(T), sizeof(T) * ((block2_cols)), DPU_XFER_DEFAULT));
	}

	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

	//print DPU content
	/*DPU_FOREACH(set,dpu,each_dpu)
    {
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}*/

	//Copy outputs from DPUS
    for(int current_line = 0; current_line < block1_rows; current_line++)
    {
    	DPU_FOREACH(set,dpu,each_dpu)
    	{
    		DPU_ASSERT(dpu_prepare_xfer(dpu, &m3[current_line*(m2_cols+padding2)+((each_dpu%NR_DPUS)*block2_cols)+((each_dpu/NR_DPUS)*block1_rows*(m2_cols+padding2))]));
    	}	
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
    for(int i = 0; i < m1_rows; i++)
    {
    	for(int j = 0; j < m2_cols; j++)
    	{
    		m3_[i*m2_cols+j] = m3[i*m2_cols+(j+i*padding2)];
    	}
    }
}

void Run_KDot_Sigmoid_vector(T *m1_, T *m2_, T *m3_, int m1_rows, int m1_cols, int m2_rows, int m2_cols, struct dpu_set_t set, struct dpu_set_t dpu, uint32_t nr_dpus, uint32_t each_dpu, char m1_name[], char m2_name[], char m3_name[], char binary_path[])
{
    
    int block1_rows = Ceil(((T)m1_rows/(NR_DPUS*NR_DPUS))/2)*2;
    int padding1 = block1_rows*(NR_DPUS*NR_DPUS)-m1_rows;
    T *m1 = (T*)malloc(sizeof(T)*(m1_rows+padding1)*(m1_cols));
    T *m3 = (T*)malloc(sizeof(T)*(m1_rows+padding1)*(m2_cols));


    for(int i = 0; i < m1_rows*m1_cols; i++)
    {
    	m1[i] = m1_[i];
    }

    for(int i = 0; i < padding1*m1_cols; i++)
    {
    	m1[(m1_rows*m1_cols)+i] = 0.0;
    }

    DPU_ASSERT(dpu_load(set, binary_path, NULL));

    DPU_FOREACH(set,dpu,each_dpu)
    {
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &m1[each_dpu* (block1_rows*m1_cols)]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m1_name, 0, sizeof(T) * ((block1_rows*m1_cols)), DPU_XFER_DEFAULT));
    
	DPU_FOREACH(set,dpu,each_dpu)
 	{
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &m2_[0]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m2_name, 0, sizeof(T) * m2_rows, DPU_XFER_DEFAULT));


	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

	//print DPU content
	/*DPU_FOREACH(set,dpu,each_dpu)
    {
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}*/

	//Copy outputs from DPUS
    DPU_FOREACH(set,dpu,each_dpu)
    {
    		
    	DPU_ASSERT(dpu_prepare_xfer(dpu, &m3[each_dpu*block1_rows]));
    }
    //printf("for ended with line: %d\n", current_line);	
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, m3_name, 0, sizeof(T) * block1_rows, DPU_XFER_DEFAULT));

    /*for(int i = 0; i < m1_rows; i++)
    {
    	for(int j = 0; j < m2_cols; j++)
    	{
    		printf("%f ",m3[i*m2_cols+j]);
    	}
    	printf("\n");
    }*/

    //remove padding and copy to m3_
    for(int i = 0; i < m1_rows; i++)
    {
    	m3_[i] = m3[i];
    }
}




int main()
{
	srand(1);
	
	T* h_X = (T*)malloc(TRAINING_SIZE*TRAINING_DIM*sizeof(T));
	for(int i = 0; i < TRAINING_SIZE*TRAINING_DIM; i++)
	{
		h_X[i] = ((T)rand()/(T)(RAND_MAX))*5.0;
	}
	
	//WEIGHTS_0
	const int W0_size = L1_SIZE*TRAINING_DIM*sizeof(T);
	T *h_W0 = (T*)malloc(W0_size);
	for(int i = 0; i < L1_SIZE*TRAINING_DIM; i++)
	{
		h_W0[i] = ((T)rand()/(T)(RAND_MAX))*1.0;
	}

	//LAYER_1
	const int L1_size = L1_SIZE*TRAINING_SIZE*sizeof(T);

	//l1 variable is the output of the first layer after first sigmoid
	T* h_layer_1 = (T*)malloc(L1_size);

	//WEIGHTS_1
	const int W1_size = L1_SIZE*1*sizeof(T);
	T *h_W1 = (T*)malloc(W1_size);
	for(int i = 0; i < L1_SIZE*1; i++)
	{
		h_W1[i] = ((T)rand()/(T)(RAND_MAX))*1.0;
	}



	//PRED AND PRED_DELTA
	const int y_size = y_dim*sizeof(T);
	T* h_pred = (T*)malloc(y_size);

 	Timer timer;
	start(&timer, 0, 0);

	//Allocate set of DPUs
	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    alloc_dpus(&set, &nr_dpus, NR_DPUS*NR_DPUS);


	Run_KDot_Sigmoid(h_X, h_W0, h_layer_1, TRAINING_SIZE, TRAINING_DIM, TRAINING_DIM, L1_SIZE, set, dpu, nr_dpus, each_dpu, "d_X", "d_W0", "d_l1", KDOT_SIGMOID_BINARY);
	Run_KDot_Sigmoid_vector(h_layer_1, h_W1, h_pred, TRAINING_SIZE, L1_SIZE, L1_SIZE, 1, set, dpu, nr_dpus, each_dpu, "d_l1", "d_W1", "d_pred", KDOT_SIGMOID2_BINARY);

  	stop(&timer, 0);
	printf("MLP Upmem Version Time (ms): ");
	print(&timer, 0, 1);

	/*printf("\n\nPrediction values are: \n");
	for(int i = 0; i < TRAINING_SIZE; i++)
	{
		printf("%d: %f\n",i, h_pred[i]);
	}
	printf("\n\n");*/
	return 0;
}


