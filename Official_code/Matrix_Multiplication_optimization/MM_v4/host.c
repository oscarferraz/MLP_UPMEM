#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include <math.h>
#include "common/common.h"
#include "common/timer.h"


#define KDOT_BINARY "KDOT"
int Ceil(float a){
    return(a > (int) a) ? (a+1) : (a);
}


int alloc_dpus(struct dpu_set_t *set, uint32_t *nr_dpus)
{	
	DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, set));
	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
	printf("Number of DPUs allocated=%d\n", nr_dpus[0]);


	printf("Test=%d\n", Ceil((float)((M1_ROWS*(M2_COLS+(M2_COLS%WORKLOAD_CONFIG))))/ (float)(WORKLOAD_CONFIG)));
	


	if(WORKLOAD_CONFIG != Ceil((float)(L1_SIZE*TRAINING_SIZE)/ (float)(MAX_DPUS))){
		printf("Workload balance setup=%d\nChange #define WORKLOAD_CONFIG to %d\n\n", Ceil((float)(M1_ROWS*M2_COLS)/ (float)(MAX_DPUS)), Ceil((float)(M1_ROWS*M2_COLS)/ (float)(MAX_DPUS)));
		return 0;
	}


	
	if(M1_ROWS >= M2_COLS){
		if(NR_DPUS != Ceil((float)(M1_ROWS*(M2_COLS+(M2_COLS%WORKLOAD_CONFIG)))/ (float)(WORKLOAD_CONFIG))){
			printf("Change #define NR_DPUS to %d\n",  Ceil((float)((M1_ROWS*(M2_COLS+(M2_COLS%WORKLOAD_CONFIG))))/ (float)(WORKLOAD_CONFIG)));
			return 0;
		}
	}else{
		if(NR_DPUS !=  Ceil((float)((M1_ROWS+(M1_ROWS%WORKLOAD_CONFIG))*M2_COLS)/ (float)(WORKLOAD_CONFIG))){
			printf("Change #define NR_DPUS to %d\n",  Ceil((float)((M1_ROWS+(M1_ROWS%WORKLOAD_CONFIG))*M2_COLS)/ (float)(WORKLOAD_CONFIG)));
			return 0;
		}
	}
		
	



	/* if(NR_DPUS != Ceil((float)(M1_ROWS*M2_COLS)/ (float)(WORKLOAD_CONFIG))){
		printf("Change #define NR_DPUS to %d\n",  Ceil((float)(M1_ROWS*M2_COLS)/ (float)(WORKLOAD_CONFIG)));
		return 0;
	} */

	if(M1_ROWS >= M2_COLS){
		if((M1_ROWS+WORKLOAD_CONFIG*M2_COLS+WORKLOAD_CONFIG)*sizeof(T) < 64*1024){
			if(FITS_IN_WRAM != 1){
				printf("The buffers fit in WRAM\nChange #define FITS_IN_WRAM to 1\n");
				return 0;
			}
		}else{
			if(FITS_IN_WRAM != 0){
				printf("All the buffers do not fit in WRAM\nChange #define FITS_IN_WRAM to 0\n");
				return 0;
			}
		}
	}else{
		if((WORKLOAD_CONFIG*M1_ROWS+M2_COLS+WORKLOAD_CONFIG)*sizeof(T) < 64*1024){
			if(FITS_IN_WRAM != 1){
				printf("The buffers fit in WRAM\nChange #define FITS_IN_WRAM to 1\n");
				return 0;
			}
		}else{
			if(FITS_IN_WRAM != 0){
				printf("All the buffers do not fit in WRAM\nChange #define FITS_IN_WRAM to 0\n");
				return 0;
			}
		}
	}


	if(FITS_IN_WRAM != 1){
		if(((M1_ROWS)*sizeof(T) >  64*1024) && ((M2_COLS)*sizeof(T) >  64*1024)){
			if(WRAM_CONFIG != 0){
				printf("The buffers do not fit in WRAM\nChange #define WRAM_CONFIG to 0\n");
				return 0;
			}
		}else{
			if(M1_ROWS >= M2_COLS){
				if ((M1_ROWS)*sizeof(T) <  64*1024){
					if(WRAM_CONFIG != 1){
						printf("M1 fits in WRAM\nChange #define WRAM_CONFIG to 1\n");
						return 0;
					}
				}else if((M2_COLS)*sizeof(T) <  64*1024){
					if(WRAM_CONFIG != 2){
						printf("M2 fits in WRAM\nChange #define WRAM_CONFIG to 2\n");
						return 0;
					}
				}
				
			}else{
				if ((M2_COLS)*sizeof(T) <  64*1024){
					if(WRAM_CONFIG != 2){
						printf("M2 fits in WRAM\nChange #define WRAM_CONFIG to 2\n");
						return 0;
					}
				}else if((M1_ROWS)*sizeof(T) <  64*1024){
					if(WRAM_CONFIG != 1){
						printf("M1 fits in WRAM\nChange #define WRAM_CONFIG to 1\n");
						return 0;
					}
				}

			}
		}
	}

	return 1;
}

void Run_KDot(T *m1, T *m2, T *m3, struct dpu_set_t set, struct dpu_set_t dpu, uint32_t nr_dpus, uint32_t each_dpu, char m1_name[], char m2_name[], char m3_name[], char binary_path[])
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

	#if WRAM_CONFIG == 0  || WRAM_CONFIG == 1
		int padding2 = M2_COLS%WORKLOAD_CONFIG;
		// printf("Number of DPUs padding=%d\n", padding2);
		T *m1_ = (T*)malloc(M1_ROWS*M1_COLS*sizeof(T));
		T *m2_ = (T*)malloc(M2_ROWS*(M2_COLS+padding2)*sizeof(T));
		T *m3_ = (T*)malloc(M1_ROWS*(M2_COLS+padding2)*sizeof(T));

		for(int i = 0; i < M1_ROWS*M1_COLS; i++){
			m1_[i] = m1[i];	
		}


		/* for(int i = 0; i < M2_ROWS; i++){
			for(int j = 0; j < M2_COLS+padding2; j++){
				if(j>=M2_COLS){
					m2_[i*(M2_COLS+padding2)+j] = 0.0;	
				}else{
					// m2_[i*(M2_COLS+padding2)+j] = m2[i*(M2_COLS)+j];	
					m2_[i*(M2_COLS+padding2)+j] = m2[i*(M2_COLS)+j];	
				}
				printf("m2_[%d]=%f", i*(M2_COLS+padding2)+j, m2_[i*(M2_COLS+padding2)+j]);
			}
			printf("\n");
			
		}
		printf("\n");
		printf("\n"); */

		for(int i = 0; i < M2_COLS+padding2; i++){
			for(int j = 0; j < M2_ROWS; j++){
				if(i*M2_ROWS+j>=M2_ROWS*M2_COLS){
					m2_[i*M2_ROWS+j] = 0.0;	
				}else{
					// m2_[i*(M2_COLS+padding2)+j] = m2[i*(M2_COLS)+j];	
					m2_[i*M2_ROWS+j] = m2[j*(M2_COLS)+i];	
				}
				// printf("m2_[%d]=%f", i*M2_ROWS+j, m2_[i*M2_ROWS+j]);
			}
			// printf("\n");
			
		}
		// printf("\n");
		// printf("\n");
	#endif



	#if WRAM_CONFIG == 2
		int padding1 = M1_ROWS%WORKLOAD_CONFIG;
		// printf("Number of DPUs padding=%d\n", padding1);
		T *m1_ = (T*)malloc((M1_ROWS+padding1)*M1_COLS*sizeof(T));
		T *m2_ = (T*)malloc(M2_ROWS*M2_COLS*sizeof(T));
		T *m3_ = (T*)malloc((M1_ROWS+padding1)*M2_COLS*sizeof(T));

		for(int i = 0; i < (M1_ROWS+padding1)*M1_COLS; i++){
			if(i>=M1_ROWS*M1_COLS){
				m1_[i] = 0.0;	
			}else{
				m1_[i] = m1[i];	
			}
			// printf("m1_[%d]=%f\n", i, m1_[i]);
		}

		for(int i = 0; i < M2_ROWS*M2_COLS; i++){
			m2_[i] = m2[i];	
		}
		
	#endif

   /*  int block1_rows = Ceil((T)m1_rows/NR_DPUS);
    int padding1 = block1_rows*NR_DPUS-m1_rows;
	int block2_rows = Ceil((((T)m2_rows/NR_DPUS)/2))*2;
    int padding2 = block2_rows*NR_DPUS-m2_rows; */
    //printf("padding2: %d\n", padding2);
    // T *m3 = (T*)malloc(sizeof(T)*(m1_rows+padding1)*(m2_rows+padding2));
 
	/* for(int i = 0; i < M1_ROWS*M1_COLS; i++){
		printf("m1[%d]=%f\n", i, m1[i]);
    } */

    /* for(int i = 0; i < m1_rows*m1_cols; i++){
    	m1[i] = m1_[i];
		// printf("m1[%d]=%f\n", i, m1[i]);
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
    } */

	

    //printf("You entered: %s\n.", binary_path);
    DPU_ASSERT(dpu_load(set, binary_path, NULL));


	#if WRAM_CONFIG == 0  || WRAM_CONFIG == 1
	DPU_FOREACH(set,dpu,each_dpu){
		// printf("number[%d]=%d\n",each_dpu, ((int)((float)(each_dpu)/(float)((M2_COLS+padding2)/WORKLOAD_CONFIG)))*M1_COLS);
		DPU_ASSERT(dpu_prepare_xfer(dpu, &m1_[((int)((float)(each_dpu)/(float)((M2_COLS+padding2)/WORKLOAD_CONFIG)))*M1_COLS]));
    }
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m1_name, 0, sizeof(T) * M1_COLS, DPU_XFER_DEFAULT));
	#endif


	#if WRAM_CONFIG == 2
    DPU_FOREACH(set,dpu,each_dpu){
			// printf("number[%d]=%d\n",each_dpu, ((int)((float)(each_dpu)/(float)((M1_ROWS+padding1)/WORKLOAD_CONFIG)))*M2_COLS);
			DPU_ASSERT(dpu_prepare_xfer(dpu, &m1_[((int)((float)(each_dpu)/(float)((M1_ROWS+padding1)/WORKLOAD_CONFIG)))*M2_COLS]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m1_name, 0, sizeof(T) * M1_COLS, DPU_XFER_DEFAULT));
	#endif

	#if WRAM_CONFIG == 0  || WRAM_CONFIG == 1
	DPU_FOREACH(set,dpu,each_dpu){
		// printf("number[%d]=%d\n",each_dpu, (each_dpu*WORKLOAD_CONFIG)%(M2_COLS+padding2)*M2_ROWS);
		DPU_ASSERT(dpu_prepare_xfer(dpu, &m2_[(each_dpu*WORKLOAD_CONFIG)%(M2_COLS+padding2)*M2_ROWS]));
    }
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m2_name, 0 , sizeof(T) * M2_ROWS*WORKLOAD_CONFIG, DPU_XFER_DEFAULT));
	#endif


	#if WRAM_CONFIG == 2
    DPU_FOREACH(set,dpu,each_dpu){
			// printf("number[%d]=%d\n",each_dpu, ((int)((float)(each_dpu)/(float)((M1_ROWS+padding1)/WORKLOAD_CONFIG)))*M2_COLS);
			DPU_ASSERT(dpu_prepare_xfer(dpu, &m1_[((int)((float)(each_dpu)/(float)((M1_ROWS+padding1)/WORKLOAD_CONFIG)))*M2_COLS]));
    }
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, m1_name, 0, sizeof(T) * M1_COLS, DPU_XFER_DEFAULT));
	#endif




	//Run DPUs
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

	//Print outputs from DPUS
	DPU_FOREACH(set,dpu,each_dpu){
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}

	DPU_FOREACH(set,dpu,each_dpu){
    		
		DPU_ASSERT(dpu_prepare_xfer(dpu, &m3_[each_dpu*WORKLOAD_CONFIG]));
	}
	//printf("for ended with line: %d\n", current_line);	
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, m3_name, 0, sizeof(T) * WORKLOAD_CONFIG, DPU_XFER_DEFAULT));

    /* for(int current_line = 0; current_line < block1_rows; current_line++)
    {
    	DPU_FOREACH(set,dpu,each_dpu)
    	{
    		
    		DPU_ASSERT(dpu_prepare_xfer(dpu, &m3[current_line*(m2_rows+padding2)+((each_dpu%NR_DPUS)*block2_rows)+((each_dpu/NR_DPUS)*block1_rows*(m2_rows+padding2))]));
    		//printf("val is: %d \n", current_line*(test2_cols+padding2)+((each_dpu%NR_DPUS)*block2_cols)+((each_dpu/NR_DPUS)*block1_rows*(test2_cols+padding2)));
    	//DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
    	}
    	//printf("for ended with line: %d\n", current_line);	
    	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, m3_name, current_line*(block2_rows)*sizeof(T), sizeof(T) * (block2_rows), DPU_XFER_DEFAULT));
    } */

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
    /* for(int i = 0; i < M1_ROWS; i++){
    	for(int j = 0; j < M2_COLS+padding2; j++){
    		// m3[i*M2_COLS+j] = m3[j+i*(M2_COLS+padding2)];
    		printf("m3_[%d][%d]=%f ", i, j, m3_[i*(M2_COLS+padding2)+j]);
    	}
    	printf("\n");
    } */

	// printf("\n");
	// printf("\n");
	// printf("\n");

	for(int i = 0; i < M1_ROWS; i++){
		for(int j = 0; j < M2_COLS; j++){
			m3[i*M2_COLS+j] = m3_[j+i*(M2_COLS+padding2)];
			// printf("m3[%d][%d]=%f ", i, j, m3[i*(M2_COLS)+j]);
		}
		// printf("\n");
    }

   free(m1_);
   free(m2_);
   free(m3_);
}

int main()
{

	if(M1_COLS!=M2_ROWS){
		printf("Matrix dimensions do not match\n");
		return 0;
	}
	
	T *m1 = (T*)malloc(M1_ROWS*M1_COLS*sizeof(T));
	for(int i = 0; i < M1_ROWS*M1_COLS; i++)
	{
		m1[i] = ((float)i);
	}
	
	//WEIGHTS_0
	T *m2 = (T*)malloc(M2_ROWS*M2_COLS*sizeof(T));
	for(int i = 0; i < M2_ROWS*M2_COLS; i++)
	{
		m2[i] = ((float)i);
	}

	//LAYER_1, LAYER_1_DELTA AND BUFFER OF LAYER 1 SIZE
	//const long signed int res_size = M1_ROWS*M2_ROWS*sizeof(T);

	//l1 variable is the output of the first layer after first sigmoid
	T *res = (T*)malloc(M1_ROWS*M2_ROWS*sizeof(T));



 	Timer timer;
	start(&timer, 0, 0);

	//Allocate set of DPUs
	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    if(alloc_dpus(&set, &nr_dpus)==0){
		return 0;
	}



	Run_KDot(m1, m2, res, set, dpu, nr_dpus, each_dpu, "d_m1", "d_m2", "d_res", KDOT_BINARY);
	printf("\n\n\n\n");
	for(int i = 0; i < M1_ROWS; i++)
	{
		for(int j = 0; j < M2_COLS; j++)
		{
			#if T_IS_FLOAT == 1
				printf("%.1f ", res[i*M2_COLS+j]);
			#endif
			#if T_IS_INT == 1
				printf("%d ", res[i*M2_COLS+j]);
			#endif
			#if T_IS_CHAR == 1
				printf("%c ", res[i*M2_COLS+j]);
			#endif
		}
		printf("\n");
	}
	printf("\n\n\n\n");
  	stop(&timer, 0);
	printf("GEMM Time (ms): ");
	print(&timer, 0, 1);
	printf("\n");
	/*printf("\n\nPrediction values are: \n");
	for(int i = 0; i < TRAINING_SIZE; i++)
	{
		printf("%d: %f\n",i, h_pred[i]);
	}
	printf("\n\n");*/


	free(m1);
	free(m2);
	free(res);

	printf("OK\n");
	return 0;
}


