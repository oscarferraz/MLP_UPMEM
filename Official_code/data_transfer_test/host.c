#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <errno.h>
#include "common/common.h"
#include "common/timer.h"

#define KDOT_RELU_BINARY "dpu"

int Ceil(float a)
{
    return(a > (int) a) ? (a+1) : (a);
}

int alloc_dpus(struct dpu_set_t *set, uint32_t *nr_dpus)
{
	DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, set));
	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
	printf("Number of DPUs allocated=%d\n", nr_dpus[0]);


	/* if(NR_DPUS != Ceil((float)(L1_SIZE*TRAINING_SIZE)/ (float)(MAX_DPUS))){
		printf("Change #define NR_DPUS to %d\n", Ceil((float)(L1_SIZE*TRAINING_SIZE)/ (float)(MAX_DPUS)), Ceil((float)(L1_SIZE*TRAINING_SIZE)/ (float)(MAX_DPUS)));
		return 0;
	} */

	return 1;
}

void Run_Test(T *m1, struct dpu_set_t set, struct dpu_set_t dpu, uint32_t nr_dpus, uint32_t each_dpu, char m1_name[], char binary_path[]){



    //load dpu binary file
    //printf("You entered: %s\n.", binary_path);
    DPU_ASSERT(dpu_load(set, binary_path, NULL));
    //DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));


	Timer timer;
	start(&timer, 0, 0);

	
	//send matrix1 with parallel transfers
	DPU_FOREACH(set,dpu,each_dpu)
	{
		// printf("test=%d\n", ((each_dpu*TRAINING_SIZE)/2048));
		DPU_ASSERT(dpu_prepare_xfer(dpu, &m1[0]));
	}
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m1_name, 0, sizeof(T) * SIZE, DPU_XFER_DEFAULT));

	
	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS)); 
	

	//print DPU content
	/* DPU_FOREACH(set,dpu,each_dpu){
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	} */

		

	stop(&timer, 0);
	printf("L1 time (ms): ");
	print(&timer, 0, 1);


}




int main(){
	
	
	T* h_X = (T*)malloc(SIZE*sizeof(T));
	for(int i = 0; i < SIZE; i++){
		// h_X[i] = ((T)rand()/(T)(RAND_MAX))*0.0;
		#if IS_FLOAT == 1
			h_X[i] = ((T)0.02);
		#endif
		#if IS_INT == 1
			h_X[i] = ((T)1);
		#endif
	}

	//Allocate set of DPUs
	struct dpu_set_t set, dpu;
	uint32_t nr_dpus, each_dpu;
	if(alloc_dpus(&set, &nr_dpus)==0){
		return 0;
	}

  	
  	Run_Test(h_X, set, dpu, nr_dpus, each_dpu, "d_X", KDOT_RELU_BINARY);


	
	

	// printf("MLP Upmem Version Time (ms): ");
	// print(&timer, 0, 1);

	// printf("\n\nPrediction values are: \n");
	/* for(int i = 0; i < TRAINING_SIZE; i++)
	{
		printf("Prediction[%d]: %f\n",i, h_pred[i]);
	}
	printf("\n\n"); */

	free(h_X);

	return 0;
}


