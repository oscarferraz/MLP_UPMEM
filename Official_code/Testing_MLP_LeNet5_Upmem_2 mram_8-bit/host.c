#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include "common/common.h"
#include "common/timer.h"

#define KDOT_SIGMOID_BINARY "KDOT_SIGMOID"
#define KDOT_SIGMOID2_BINARY "KDOT_SIGMOID2"
#define KDOT_SIGMOID3_BINARY "KDOT_SIGMOID3"

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

void Run_KDot_Sigmoid_L1(T *m1, T *m2, T *m3, struct dpu_set_t set, struct dpu_set_t dpu, uint32_t nr_dpus, uint32_t each_dpu, char m1_name[], char m2_name[], char m3_name[], char binary_path[]){

	//calculating block sizes to send to each dpu
	//usage of ceil and padding allow for blocks of same size
	//which is required by upmem assert functions for parallel
	//transfers
	//Note the blkocking must obey the following rules:
    //For m1:
    	//nr columns must be even
		//nr should be even but I don't think it is a must
	//For m2:
		//one of the dimensions must be the same as for m1 for multiplication to be possible.
		//the dimension that must be same depends if you want to transpose or not
	//The padding for matrix 2 since it will be the columns in the result matrix must occour in a way that it makes the number of
	//elements in the column even, that is why block2 is calculated diferently from block 1
    

    //load dpu binary file
    //printf("You entered: %s\n.", binary_path);
    DPU_ASSERT(dpu_load(set, binary_path, NULL));
    //DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));

	

	FILE *file = fopen("original.txt", "w");
	if (file == NULL) {
		perror("Error opening file");
		return;
	}

	for (int i = 0; i < TRAINING_DIM; i++) {
		for (int j = 0; j < L1_SIZE; j++) {
			#if IS_FLOAT == 1
				fprintf(file, "%f ", m2[i*(L1_SIZE)+j]);
			#endif
			#if IS_INT == 1
				fprintf(file, "%d ", m2[i*(L1_SIZE)+j]);
			#endif
		}

		
		fprintf(file, "\n");
	}

	fclose(file);

	T *m2_ = (T*)malloc(TRAINING_DIM*L1_SIZE*sizeof(T));

	for(int i = 0; i < L1_SIZE; i++){
		for(int j = 0; j < TRAINING_DIM; j++){
			// m2_[i*(M2_COLS+padding2)+j] = m2[i*(M2_COLS)+j];	
			m2_[i*TRAINING_DIM+j] = m2[j*(L1_SIZE)+i];	
		}

	}
	// printf("\n");


	FILE *file2 = fopen("transposed.txt", "w");
	if (file2 == NULL) {
		perror("Error opening file");
		return;
	}

	for (int i = 0; i < L1_SIZE; i++) {
		for (int j = 0; j < TRAINING_DIM; j++) {
			#if IS_FLOAT == 1
				fprintf(file2, "%f ", m2_[i*(TRAINING_DIM)+j]);
			#endif
			#if IS_INT == 1
				fprintf(file2, "%d ", m2_[i*(TRAINING_DIM)+j]);
			#endif
		}
		fprintf(file2, "\n");
	}

	fclose(file2);

	Timer timer;
	start(&timer, 0, 0);

	
	//send matrix1 with parallel transfers
	DPU_FOREACH(set,dpu,each_dpu)
	{
		// printf("test=%d\n", ((each_dpu*TRAINING_SIZE)/2048));
		DPU_ASSERT(dpu_prepare_xfer(dpu, &m1[each_dpu*(TRAINING_SIZE/NR_DPUS)*TRAINING_DIM]));
	}
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m1_name, 0, sizeof(T) * (TRAINING_SIZE/NR_DPUS)*TRAINING_DIM, DPU_XFER_DEFAULT));

	//send matrix2 with parallel transfers
	DPU_FOREACH(set,dpu,each_dpu)
	{
		DPU_ASSERT(dpu_prepare_xfer(dpu, &m2_[0]));
	}
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m2_name, 0, sizeof(T) * TRAINING_DIM * L1_SIZE, DPU_XFER_DEFAULT));
		
	
	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS)); 
	

	//print DPU content
	/* DPU_FOREACH(set,dpu,each_dpu){
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}
 */
	


	
	DPU_FOREACH(set,dpu,each_dpu){
		// printf("test=%d\n", each_dpu*(L1_SIZE/2));
		DPU_ASSERT(dpu_prepare_xfer(dpu, &m3[each_dpu*(TRAINING_SIZE/NR_DPUS)*L1_SIZE]));
	}
	//printf("for ended with line: %d\n", current_line);	
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, m3_name, 0, sizeof(T) *  (TRAINING_SIZE/NR_DPUS)*L1_SIZE, DPU_XFER_DEFAULT));
		

	stop(&timer, 0);
	printf("L1 time (ms): ");
	print(&timer, 0, 1);

	FILE *file3 = fopen("result.txt", "w");
	if (file3 == NULL) {
		perror("Error opening file");
		return;
	}

	for (int i = 0; i < TRAINING_SIZE; i++) {
		for (int j = 0; j < L1_SIZE; j++) {
			#if IS_FLOAT == 1
				fprintf(file3, "%f ", m3[i*(L1_SIZE)+j]);
			#endif
			#if IS_INT == 1
				fprintf(file3, "%d ", m3[i*(L1_SIZE)+j]);
			#endif
		}
		fprintf(file3, "\n");
	}

	fclose(file3);

   free(m2_);
}

void Run_KDot_Sigmoid_L2(T *m1, T *m2, T *m3, struct dpu_set_t set, struct dpu_set_t dpu, uint32_t nr_dpus, uint32_t each_dpu, char m1_name[], char m2_name[], char m3_name[], char binary_path[]){

	//calculating block sizes to send to each dpu
	//usage of ceil and padding allow for blocks of same size
	//which is required by upmem assert functions for parallel
	//transfers
	//Note the blkocking must obey the following rules:
    //For m1:
    	//nr columns must be even
		//nr should be even but I don't think it is a must
	//For m2:
		//one of the dimensions must be the same as for m1 for multiplication to be possible.
		//the dimension that must be same depends if you want to transpose or not
	//The padding for matrix 2 since it will be the columns in the result matrix must occour in a way that it makes the number of
	//elements in the column even, that is why block2 is calculated diferently from block 1
    

    //load dpu binary file
    //printf("You entered: %s\n.", binary_path);
    DPU_ASSERT(dpu_load(set, binary_path, NULL));
    //DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));

	

	FILE *file = fopen("original_L2.txt", "w");
	if (file == NULL) {
		perror("Error opening file");
		return;
	}

	for (int i = 0; i < L1_SIZE; i++) {
		for (int j = 0; j < L2_SIZE; j++) {
			#if IS_FLOAT == 1
				fprintf(file, "%f ", m2[i*(L2_SIZE)+j]);
			#endif
			#if IS_INT == 1
				fprintf(file, "%d ", m2[i*(L2_SIZE)+j]);
			#endif
		}
		fprintf(file, "\n");
	}

	fclose(file);

	T *m2_ = (T*)malloc(L1_SIZE*L2_SIZE*sizeof(T));

	for(int i = 0; i < L2_SIZE; i++){
		for(int j = 0; j < L1_SIZE; j++){
			// m2_[i*(M2_COLS+padding2)+j] = m2[i*(M2_COLS)+j];	
			m2_[i*L1_SIZE+j] = m2[j*(L2_SIZE)+i];	
		}

	}
	// printf("\n");


	FILE *file2 = fopen("transposed_L2.txt", "w");
	if (file2 == NULL) {
		perror("Error opening file");
		return;
	}

	for (int i = 0; i < L2_SIZE; i++) {
		for (int j = 0; j < L1_SIZE; j++) {
			#if IS_FLOAT == 1
				fprintf(file2, "%f ", m2_[i*(L1_SIZE)+j]);
			#endif
			#if IS_INT == 1
				fprintf(file2, "%d ", m2_[i*(L1_SIZE)+j]);
			#endif
		}
		fprintf(file2, "\n");
	}

	fclose(file2);

	Timer timer;
	start(&timer, 0, 0);

	//send matrix1 with parallel transfers
	DPU_FOREACH(set,dpu,each_dpu)
	{
		// printf("test=%d\n", ((each_dpu*TRAINING_SIZE)/2048));
		DPU_ASSERT(dpu_prepare_xfer(dpu, &m1[each_dpu*(TRAINING_SIZE/NR_DPUS)*L1_SIZE]));
	}
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m1_name, 0, sizeof(T) * (TRAINING_SIZE/NR_DPUS)*L1_SIZE, DPU_XFER_DEFAULT));

	//send matrix2 with parallel transfers
	DPU_FOREACH(set,dpu,each_dpu)
	{
		DPU_ASSERT(dpu_prepare_xfer(dpu, &m2_[0]));
	}
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m2_name, 0, sizeof(T) * L2_SIZE * L1_SIZE, DPU_XFER_DEFAULT));

		
	
	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS)); 
	

	//print DPU content
	/* DPU_FOREACH(set,dpu,each_dpu){
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	} */

	

	


	DPU_FOREACH(set,dpu,each_dpu){
		DPU_ASSERT(dpu_prepare_xfer(dpu, &m3[each_dpu*(TRAINING_SIZE/NR_DPUS)*L2_SIZE]));
	}
	//printf("for ended with line: %d\n", current_line);	
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, m3_name, 0, sizeof(T) *  (TRAINING_SIZE/NR_DPUS)*L2_SIZE, DPU_XFER_DEFAULT));
		

	stop(&timer, 0);
	printf("L2 time (ms): ");
	print(&timer, 0, 1);

	FILE *file3 = fopen("result2.txt", "w");
	if (file3 == NULL) {
		perror("Error opening file");
		return;
	}

	for (int i = 0; i < TRAINING_SIZE; i++) {
		for (int j = 0; j < L2_SIZE; j++) {
			#if IS_FLOAT == 1
				fprintf(file3, "%f ", m3[i*(L2_SIZE)+j]);
			#endif
			#if IS_INT == 1
				fprintf(file3, "%d ", m3[i*(L2_SIZE)+j]);
			#endif
		}
		fprintf(file3, "\n");
	}

	fclose(file3);

   free(m2_);
}


void Run_KDot_Sigmoid_Output(T *m1, T *m2, T *m3, struct dpu_set_t set, struct dpu_set_t dpu, uint32_t nr_dpus, uint32_t each_dpu, char m1_name[], char m2_name[], char m3_name[], char binary_path[]){

	//calculating block sizes to send to each dpu
	//usage of ceil and padding allow for blocks of same size
	//which is required by upmem assert functions for parallel
	//transfers
	//Note the blkocking must obey the following rules:
    //For m1:
    	//nr columns must be even
		//nr should be even but I don't think it is a must
	//For m2:
		//one of the dimensions must be the same as for m1 for multiplication to be possible.
		//the dimension that must be same depends if you want to transpose or not
	//The padding for matrix 2 since it will be the columns in the result matrix must occour in a way that it makes the number of
	//elements in the column even, that is why block2 is calculated diferently from block 1
    

    //load dpu binary file
    //printf("You entered: %s\n.", binary_path);
    DPU_ASSERT(dpu_load(set, binary_path, NULL));
    //DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));

	

	FILE *file = fopen("original_L3.txt", "w");
	if (file == NULL) {
		perror("Error opening file");
		return;
	}



	for (int i = 0; i < L2_SIZE; i++) {
		for (int j = 0; j < OUTPUT_SIZE; j++) {
			#if IS_FLOAT == 1
				fprintf(file, "%f ", m2[i*(OUTPUT_SIZE)+j]);
			#endif
			#if IS_INT == 1
				fprintf(file, "%d ", m2[i*(OUTPUT_SIZE)+j]);
			#endif
		}
		fprintf(file, "\n");
	}

	fclose(file);

	T *m2_ = (T*)malloc(L2_SIZE*OUTPUT_SIZE*sizeof(T));

	for(int i = 0; i < OUTPUT_SIZE; i++){
		for(int j = 0; j < L2_SIZE; j++){
			// m2_[i*(M2_COLS+padding2)+j] = m2[i*(M2_COLS)+j];	
			m2_[i*L2_SIZE+j] = m2[j*(OUTPUT_SIZE)+i];	
		}

	}
	// printf("\n");


	FILE *file2 = fopen("transposed_L3.txt", "w");
	if (file2 == NULL) {
		perror("Error opening file");
		return;
	}

	for (int i = 0; i < OUTPUT_SIZE; i++) {
		for (int j = 0; j < L2_SIZE; j++) {
			#if IS_FLOAT == 1
				fprintf(file2, "%f ", m2_[i*(L2_SIZE)+j]);
			#endif
			#if IS_INT == 1
				fprintf(file2, "%d ", m2_[i*(L2_SIZE)+j]);
			#endif
		}
		fprintf(file2, "\n");
	}

	fclose(file2);

	Timer timer;
	start(&timer, 0, 0);
	//send matrix1 with parallel transfers
	DPU_FOREACH(set,dpu,each_dpu)
	{
		// printf("test=%d\n", ((each_dpu*TRAINING_SIZE)/2048));
		DPU_ASSERT(dpu_prepare_xfer(dpu, &m1[each_dpu*(TRAINING_SIZE/NR_DPUS)*L2_SIZE]));
	}
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m1_name, 0, sizeof(T) * (TRAINING_SIZE/NR_DPUS)*L2_SIZE, DPU_XFER_DEFAULT));

	//send matrix2 with parallel transfers
	DPU_FOREACH(set,dpu,each_dpu)
	{
		DPU_ASSERT(dpu_prepare_xfer(dpu, &m2_[0]));
	}
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m2_name, 0, sizeof(T) * OUTPUT_SIZE * L2_SIZE, DPU_XFER_DEFAULT));



		
	
	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS)); 
	

	//print DPU content
	/* DPU_FOREACH(set,dpu,each_dpu){
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	} */


	DPU_FOREACH(set,dpu,each_dpu){

		DPU_ASSERT(dpu_prepare_xfer(dpu, &m3[each_dpu * 8 * OUTPUT_SIZE]));
	}
	//printf("for ended with line: %d\n", current_line);	
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, m3_name, 0, sizeof(T) * 8 * OUTPUT_SIZE, DPU_XFER_DEFAULT));

	
	stop(&timer, 0);
	printf("L3 time (ms): ");
	print(&timer, 0, 1);
	printf("\n");
	


	FILE *file3 = fopen("result3.txt", "w");
	if (file3 == NULL) {
		perror("Error opening file");
		return;
	}


	for (int i = 0; i < NR_DPUS*8; i++) {
			for (int j = 0; j < OUTPUT_SIZE; j++) {
				#if IS_FLOAT == 1
					fprintf(file3, "%f ", m3[i*(OUTPUT_SIZE)+j]);
				#endif
				#if IS_INT == 1
					fprintf(file3, "%d ", m3[i*(OUTPUT_SIZE)+j]);
				#endif
			}
			fprintf(file3, "\n");
		}

	fclose(file3);

   free(m2_);
}




int main(){

	printf("L1 size=%lu bytes\n", sizeof(T)*(TRAINING_SIZE/NR_DPUS)*TRAINING_DIM + sizeof(T)*TRAINING_DIM*L1_SIZE + sizeof(T)*(TRAINING_SIZE/NR_DPUS)*L1_SIZE );
	printf("L2 size=%lu bytes\n", sizeof(T)*(TRAINING_SIZE/NR_DPUS)*L1_SIZE + sizeof(T)*L1_SIZE*L2_SIZE + sizeof(T)*(TRAINING_SIZE/NR_DPUS)*L2_SIZE );
	printf("L3 size=%lu bytes\n", sizeof(T)*(TRAINING_SIZE/NR_DPUS)*L2_SIZE + sizeof(T)*L2_SIZE*OUTPUT_SIZE + sizeof(T)*(TRAINING_SIZE/NR_DPUS)*OUTPUT_SIZE );
	
	
	T* h_X = (T*)malloc(TRAINING_SIZE*TRAINING_DIM*sizeof(T));
	for(int i = 0; i < TRAINING_SIZE*TRAINING_DIM; i++){
		// h_X[i] = ((T)rand()/(T)(RAND_MAX))*0.0;
		#if IS_FLOAT == 1
			h_X[i] = ((T)0.02);
		#endif
		#if IS_INT == 1
			h_X[i] = ((T)1);
		#endif
	}
	
	//WEIGHTS_0
	size_t W0_size = TRAINING_DIM*L1_SIZE*sizeof(T);
	T *h_W0 = (T*)malloc(W0_size);
	for(int i = 0; i < TRAINING_DIM*L1_SIZE; i++){
		// h_X[i] = ((T)rand()/(T)(RAND_MAX))*0.0;
		#if IS_FLOAT == 1
			h_W0[i] = ((T)0.03);
		#endif
		#if IS_INT == 1
			h_W0[i] = ((T)1);
		#endif
	}
	
	//LAYER_1
	size_t L1_size = TRAINING_SIZE*L1_SIZE*sizeof(T);

	//output of previous layer
	T* h_layer_1 = (T*)malloc(L1_size);

	//WEIGHTS_1
	size_t W1_size = L1_SIZE*L2_SIZE*sizeof(T);
	T *h_W1 = (T*)malloc(W1_size);
	for(int i = 0; i < L1_SIZE*L2_SIZE; i++){
		// h_W1[i] = ((T)rand()/(T)(RAND_MAX))*0.0;
		#if IS_FLOAT == 1
			h_W1[i] = ((T)0.04);
		#endif
		#if IS_INT == 1
			h_W1[i] = ((T)2);
		#endif
	}

	//LAYER_2
	size_t L2_size = L2_SIZE*TRAINING_SIZE*sizeof(T);

	//output of previous layer
	T* h_layer_2 = (T*)malloc(L2_size);

	//WEIGHTS_2
	size_t W2_size = L2_SIZE*OUTPUT_SIZE*sizeof(T);
	T *h_W2 = (T*)malloc(W2_size);
	for(int i = 0; i < L2_SIZE*OUTPUT_SIZE; i++){
		// h_W2[i] = ((T)rand()/(T)(RAND_MAX))*0.0;
		#if IS_FLOAT == 1
			h_W2[i] = ((T)0.01);
		#endif
		#if IS_INT == 1
			h_W2[i] = ((T)2);
		#endif
	}


	//PRED AND PRED_DELTA
	size_t y_size = NR_DPUS*8*sizeof(T);
	
	T* h_pred = (T*)malloc(y_size);

 	Timer timer;
	int warmup = 0;
	int reps = 1;

	//Allocate set of DPUs
	struct dpu_set_t set, dpu;
	uint32_t nr_dpus, each_dpu;
	if(alloc_dpus(&set, &nr_dpus)==0){
		return 0;
	}

  	
  	Run_KDot_Sigmoid_L1(h_X, h_W0, h_layer_1, set, dpu, nr_dpus, each_dpu, "d_X", "d_W0", "d_l1", KDOT_SIGMOID_BINARY);
	Run_KDot_Sigmoid_L2(h_layer_1, h_W1, h_layer_2, set, dpu, nr_dpus, each_dpu, "d_l1", "d_W1", "d_l2", KDOT_SIGMOID2_BINARY);
	Run_KDot_Sigmoid_Output(h_layer_2, h_W2, h_pred, set, dpu, nr_dpus, each_dpu, "d_l2", "d_W2", "d_pred", KDOT_SIGMOID3_BINARY);


	
	

	// printf("MLP Upmem Version Time (ms): ");
	// print(&timer, 0, 1);

	// printf("\n\nPrediction values are: \n");
	/* for(int i = 0; i < TRAINING_SIZE; i++)
	{
		printf("Prediction[%d]: %f\n",i, h_pred[i]);
	}
	printf("\n\n"); */

	free(h_X);
	free(h_W0);
	free(h_layer_1);
	free(h_W1);
	free(h_layer_2);
	free(h_W2);
	free(h_pred);

	return 0;
}


