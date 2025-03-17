#include <stdio.h>
#include <stdlib.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"



BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization


#if T == float
	static union{
		float f;
		struct {
			int i;
		} n;
	} _eco;

	#define EXP_A 12102203  // Adjusted for float precision
	#define EXP_C 127 * (1 << 23) - 60801  // Adjusted bias
	#define exponential(y) (_eco.n.i = EXP_A * (y) + EXP_C, _eco.f)
#endif


/* static union
{
    double d;
    struct {
      int j,i;
    } n;
 } _eco;

#define EXP_A 1512775
#define EXP_C 68243 /*see text for choice of values */
/*#define exponential(y) (_eco.n.i=EXP_A*(y)+(1072693248-EXP_C), _eco.d) */



#if GEMM == 1
	void  kDot_m1_m2T_L1( __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output){
		for(int i = 0; i < (TRAINING_SIZE/NR_DPUS); i++){
			for(int j = me()*L1_SIZE/NR_TASKLETS; j < me()*(L1_SIZE/NR_TASKLETS)+(L1_SIZE/NR_TASKLETS); j++){
				#if IS_FLOAT == 1
					output[i*L1_SIZE+j]=0.0;
				#endif
				#if IS_INT == 1
					output[i*L1_SIZE+j]=0;
				#endif
				for(int k = 0; k < TRAINING_DIM; k++){
					output[i*L1_SIZE+j]+=m1[k]*m2[j*TRAINING_DIM+k];
					/* if(me()==0){
						printf("output[%d]=%f, m1[%d]=%f, m2[%d*TRAINING_DIM+%d]=%f\n", j,  output[j], k, m1[k], j, k, m2[j*TRAINING_DIM+k]);
					} */
				}
				#if IS_FLOAT == 1
					output[i*L1_SIZE+j]= 1.0 / (1.0 + exponential(-output[i*L1_SIZE+j]));
				#endif
				#if IS_INT == 1
					output[i*L1_SIZE+j]=(output[i*L1_SIZE+j]*256)/(256+abs(output[i*L1_SIZE+j]));
				#endif
				// printf("output[%d]=%f\n", i*TRAINING_DIM+j, output[i*TRAINING_DIM+j]);
			}
		}
	}

	void  kDot_m1_m2T_L2( __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output){
		for(int i = 0; i < (TRAINING_SIZE/NR_DPUS); i++){
			for(int j = me()*L2_SIZE/NR_TASKLETS; j <  me()*(L2_SIZE/NR_TASKLETS)+(L2_SIZE/NR_TASKLETS); j++){
				#if IS_FLOAT == 1
					output[i*L2_SIZE+j]=0.0;
				#endif
				#if IS_INT == 1
					output[i*L2_SIZE+j]=0;
				#endif
				for(int k = 0; k < L1_SIZE; k++){
					output[i*L2_SIZE+j]+=m1[k]*m2[j*L1_SIZE+k];

					/* if(me()==0){
						printf("output[%d]=%f, m1[%d]=%f, m2[%d*L1_SIZE+%d]=%f\n", j,  output[j], k, m1[k], j, k, m2[j*L1_SIZE+k]);
					} */
				}
				#if IS_FLOAT == 1
					output[i*L2_SIZE+j]= 1.0 / (1.0 + exponential(-output[i*L2_SIZE+j]));
				#endif
				#if IS_INT == 1
					output[i*L2_SIZE+j]=(output[i*L2_SIZE+j]*256)/(256+abs(output[i*L2_SIZE+j]));
				#endif
				/* if(me()==0){
					printf("output[%d]=%f\n", j,  output[j]);
				} */

				// printf("output[%d]=%f\n", i*TRAINING_DIM+j, output[i*TRAINING_DIM+j]);
				// printf("thread=%d\n", me());
			}
		}
	}

	#if OUTPUT_SIZE >= NR_TASKLETS
		void  kDot_m1_m2T_Output( __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output){
			// for(int i = 0; i < (TRAINING_SIZE/NR_DPUS); i++){
				for(int j = me()*OUTPUT_SIZE/16; j <  me()*(OUTPUT_SIZE/16)+(OUTPUT_SIZE/16); j++){
					output[j]=0.0;
					for(int k = 0; k < L2_SIZE; k++){
						output[j]+=m1[k]*m2[j*L2_SIZE+k];

						/* if(me()==0){
							printf("output[%d]=%f, m1[%d]=%f, m2[%d*L1_SIZE+%d]=%f\n", j,  output[j], k, m1[k], j, k, m2[j*L1_SIZE+k]);
						} */
					}
					output[j]= 1.0 / (1.0 + exponential(-output[j]));
					/* if(me()==0){
						printf("output[%d]=%f\n", j,  output[j]);
					} */

					// printf("output[%d]=%f\n", i*TRAINING_DIM+j, output[i*TRAINING_DIM+j]);
					// printf("thread=%d\n", me());
				}
			// }
		}
	#else
		void  kDot_m1_m2T_Output( __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output, __dma_aligned T *sum){
			barrier_t *barr_addr = &my_barrier;
			for(int i = 0; i < (TRAINING_SIZE/NR_DPUS); i++){
				sum[me()]=0;
				barrier_wait(barr_addr);
				for(int k = me()*L2_SIZE/NR_TASKLETS; k < me()*(L2_SIZE/NR_TASKLETS)+(L2_SIZE/NR_TASKLETS); k++){
					sum[me()]+=m1[k]*m2[k];
				}

				
				barrier_wait(barr_addr);
				/* if(me()==0){
					printf("sum[%d]=%f\n", me(),  sum[me()]);
				} */

				for(int k = 1; k < NR_TASKLETS; k*=2){
					int index= 2*k*me();
					
					if(index < NR_TASKLETS){
						// printf("thread=%d, index=%d, k=%d\n", me(), index, k);
						// printf("\n");
						sum[index]+=sum[index+k];
						/* if(me()==1){
							printf("output[%d]=%f, index=%d\n", me(),  sum[me()], index+k);
						} */
					}
					barrier_wait(barr_addr);
				}

				/* barrier_wait(barr_addr);

				if(me()==0){
					printf("sum[%d]=%f\n", me(),  sum[me()]);
				}
				
				barrier_wait(barr_addr); */

				if(me()==0){
					#if IS_FLOAT == 1
						output[i]= 1.0 / (1.0 + exponential(-sum[0]));
					#endif
					#if IS_INT == 1
						output[i]=(sum[0]*256)/(256+abs(sum[0]));
					#endif
				}
				/* if(me()==0){
					printf("output[%d]=%f\n", me(),  output[me()]);
				} */
			}
		}
	#endif
#else
	void  kDot_m1_m2T_L1( __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output){
		for(int j = me()*L1_SIZE/NR_TASKLETS; j < me()*(L1_SIZE/NR_TASKLETS)+(L1_SIZE/NR_TASKLETS); j++){
			#if IS_FLOAT == 1
				output[j]=0.0;
			#endif
			#if IS_INT == 1
				output[j]=0;
			#endif
			for(int k = 0; k < TRAINING_DIM; k++){
				output[j]+=m1[k]*m2[j*TRAINING_DIM+k];
				// if(me()==0){
				// 	printf("output[%d]=%f, m1[%d]=%f, m2[%d*TRAINING_DIM+%d]=%f\n", j,  output[j], k, m1[k], j, k, m2[j*TRAINING_DIM+k]);
				// }
			}
			#if IS_FLOAT == 1
				output[j]= 1.0 / (1.0 + exponential(-output[j]));
			#endif
			#if IS_INT == 1
				output[j]=(output[j]*256)/(256+abs(output[j]));
			#endif
			
			// printf("output[%d]=%f\n", j, output[j]);
		}
	}
	void  kDot_m1_m2T_L2( __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output){
		for(int j = me()*L2_SIZE/NR_TASKLETS; j <  me()*(L2_SIZE/NR_TASKLETS)+(L2_SIZE/NR_TASKLETS); j++){
			#if IS_FLOAT == 1
				output[j]=0.0;
			#endif
			#if IS_INT == 1
				output[j]=0;
			#endif
			for(int k = 0; k < L1_SIZE; k++){
				output[j]+=m1[k]*m2[j*L1_SIZE+k];

				/* if(me()==0){
					printf("output[%d]=%f, m1[%d]=%f, m2[%d*L1_SIZE+%d]=%f\n", j,  output[j], k, m1[k], j, k, m2[j*L1_SIZE+k]);
				} */
			}
			#if IS_FLOAT == 1
				output[j]= 1.0 / (1.0 + exponential(-output[j]));
			#endif
			#if IS_INT == 1
				output[j]=(output[j]*256)/(256+abs(output[j]));
			#endif
			/* if(me()==0){
				printf("output[%d]=%f\n", j,  output[j]);
			} */

			// printf("output[%d]=%f\n", i*TRAINING_DIM+j, output[i*TRAINING_DIM+j]);
			// printf("thread=%d\n", me());
		}
	}


	#if OUTPUT_SIZE >= 16
		void  kDot_m1_m2T_Output( __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output){
			// for(int i = 0; i < (TRAINING_SIZE/NR_DPUS); i++){
				for(int j = me()*OUTPUT_SIZE/16; j <  me()*(OUTPUT_SIZE/16)+(OUTPUT_SIZE/16); j++){
					output[2*j]=0.0;
					for(int k = 0; k < L2_SIZE; k++){
						output[j]+=m1[k]*m2[j*L2_SIZE+k];

						/* if(me()==0){
							printf("output[%d]=%f, m1[%d]=%f, m2[%d*L1_SIZE+%d]=%f\n", j,  output[j], k, m1[k], j, k, m2[j*L1_SIZE+k]);
						} */
					}
					output[2*j]= 1.0 / (1.0 + exponential(-output[j]));
					/* if(me()==0){
						printf("output[%d]=%f\n", j,  output[j]);
					} */

					// printf("output[%d]=%f\n", i*TRAINING_DIM+j, output[i*TRAINING_DIM+j]);
					// printf("thread=%d\n", me());
				}
			// }
		}
	#else
		void  kDot_m1_m2T_Output( __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output, __dma_aligned T *sum){
			barrier_t *barr_addr = &my_barrier;
			// for(int i = 0; i < (TRAINING_SIZE/NR_DPUS); i++){
				// for(int j = me()*OUTPUT_SIZE/16; j <  me()*(OUTPUT_SIZE/16)+(OUTPUT_SIZE/16); j++){
				// if(me()==0){
			for(int k = me()*L2_SIZE/NR_TASKLETS; k < me()*(L2_SIZE/NR_TASKLETS)+(L2_SIZE/NR_TASKLETS); k++){
				sum[me()]+=m1[k]*m2[k];
			}

			// if(me()==0){
			// 	printf("sum[%d]=%f\n", me(),  sum[me()]);
			// }
			barrier_wait(barr_addr);

			for(int k = 1; k < NR_TASKLETS; k*=2){
				int index= 2*k*me();
				
				if(index < NR_TASKLETS){
					// printf("thread=%d, index=%d, k=%d\n", me(), index, k);
					// printf("\n");
					sum[index]+=sum[index+k];
					/* if(me()==1){
						printf("output[%d]=%f, index=%d\n", me(),  sum[me()], index+k);
					} */
				}
				barrier_wait(barr_addr);
			}

			/* barrier_wait(barr_addr);

			if(me()==0){
				printf("sum[%d]=%f\n", me(),  sum[me()]);
			}
			
			barrier_wait(barr_addr); */

			if(me()==0){
				#if IS_FLOAT == 1
					output[0]= 1.0 / (1.0 + exponential(-sum[0]));
				#endif
				#if IS_INT == 1
					output[0]=(sum[0]*256)/(256+abs(sum[0]));
				#endif
			}
			/* if(me()==0){
				printf("output[%d]=%f\n", me(),  output[me()]);
			} */
		}
	#endif
#endif

