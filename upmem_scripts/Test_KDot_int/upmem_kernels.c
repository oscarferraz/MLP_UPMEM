#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

//#include "math.h"

#define NR_ELEM_PER_TASKLETS (TRAINING_SIZE*TRAINING_DIM/NR_TASKLETS)
#define CACHE_SIZE 768

__mram_ptr int* kDot(const int n_threads, const int cache_size, __mram_ptr int const *m1, __mram_ptr int const *m2, __mram_ptr int *output, __dma_aligned int cache1[CACHE_SIZE], __dma_aligned int cache2[CACHE_SIZE], const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{	

	int max_threads = m1_rows * m2_columns / 2;
	int t_output[max_threads];
	int n_elem = m1_rows * m2_columns / n_threads;
	printf("nelem is: %d\n", n_threads);
	if(n_threads > max_threads)
	{
		n_elem = m1_rows * m2_columns / max_threads;
	}

	if(me() < max_threads)
	{
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		//printf("m2 cols: %d\n",m2_columns);
	    int r = (int)(i / m2_columns);
	    int c = i % m2_columns;
	    t_output[me()] = 0.0;

	    for( int k = 0; k < m1_columns; ++k )
	    {
	    	mram_read(&m1[r * m1_columns + k], &cache1[r * m1_columns + k], sizeof(int)*4);
	    	mram_read(&m2[k * m2_columns + c], &cache2[k * m2_columns + c], sizeof(int)*4);
	        t_output[me()] += cache1[r * m1_columns + k] * cache2[k * m2_columns + c];
	       

	    	//mram_read(&m1[r * m1_columns + k], &cachea[me()][0+ (k%2)], sizeof(float)*2);
	   		//mram_read(&m2[k * m2_columns + c], &cacheb[me()][0+ (k%2)], sizeof(float)*2);
	        //t_output[me()] += cachea[me()][0+ (k%2)] * cacheb[me()][0+ (k%2)];
	        //printf(" m1 I'm thread %d and Value in column %d is %.d\n",me(),k, m1[r * m1_columns + k]);
	        //printf("cachea I'm thread %d and Value in column %d is %.10f\n",me(),k, cachea[me()][0]);

	    }
	    output[i] = t_output[me()];
	    //barrier_wait(barrier_address);
	    printf("Stat is: %u. Value in column %d is %d\n",check_stack(),i, output[i]);
	}
	}
	barrier_wait(barrier_address);
	/*if(me() == 0)
	{
		for(int j = 0; j < m1_rows*m2_columns; j++)
		{
			printf("Value in column %d is %.10f\n",j, output[j]);
			//printf("%d\n",m1_columns);
		}
	}
	barrier_wait(barrier_address);*/
		
	return output;
}