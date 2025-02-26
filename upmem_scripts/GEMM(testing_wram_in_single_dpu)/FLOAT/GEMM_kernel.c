#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

int min(int a, int b)
{
	return(a > b) ? b : a;
}



__mram_ptr float* kDot_WRAM1(const int n_threads, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[BLOCK_SIZE], __dma_aligned float cache2[BLOCK_SIZE], const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{	

	int max_threads = m1_rows * m2_columns / 2;
	float t_output[max_threads];
	int n_elem = m1_rows * m2_columns / n_threads;

	if(n_threads > max_threads)
	{
		n_elem = m1_rows * m2_columns / max_threads;
	}
	//printf("n_elem value in kdot is: %d\n", n_elem);
 
	int max_read_elem = min(512, m1_rows * m1_columns);
	int max_read_elem2 = min(512, m1_columns * m2_columns);
	int iter1 = (m1_rows * m1_columns / max_read_elem);
	int iter2 = (m1_columns * m2_columns / max_read_elem);

	for(int i = 0; i < (m1_rows * m1_columns / max_read_elem) ; i++)
	{
		mram_read(&m1[i*max_read_elem], &cache1[i*max_read_elem], sizeof(float)*max_read_elem);

	}
	mram_read(&m1[iter1 * max_read_elem], &cache1[iter1 * max_read_elem], sizeof(float)*(m1_rows * m1_columns % max_read_elem));


	for(int i = 0; i < (m1_columns * m2_columns / max_read_elem) ; i++)
	{
		mram_read(&m2[i*max_read_elem2], &cache2[i*max_read_elem2], sizeof(float)*max_read_elem2);
	}
	mram_read(&m2[iter2 * max_read_elem2], &cache2[iter2 * max_read_elem2], sizeof(float)*(m1_columns * m2_columns % max_read_elem2));	
	
  
  /*for(int i = 0; i < 50*50; i++)
	{
		printf("Value %d is: %.10f\n", i, cache1[i]);
	}*/
 
 
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
        t_output[me()] += cache1[r * m1_columns + k] * cache2[k * m2_columns + c];
	       

	    	//mram_read(&m1[r * m1_columns + k], &cachea[me()][0+ (k%2)], sizeof(float)*2);
	   		//mram_read(&m2[k * m2_columns + c], &cacheb[me()][0+ (k%2)], sizeof(float)*2);
	        //t_output[me()] += cachea[me()][0+ (k%2)] * cacheb[me()][0+ (k%2)];
	        //printf(" m1 I'm thread %d and Value in column %d is %.10f\n",me(),k, m1[r * m1_columns + k]);
	        //printf("cachea I'm thread %d and Value in column %d is %.10f\n",me(),k, cachea[me()][0]);

	    }
	    output[i] = t_output[me()];
	    //barrier_wait(barrier_address);
	    //printf("Stat is: %u. Value in column %d is %.10f\n",check_stack(),i, output[i]);
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

__mram_ptr float* kDot_WRAM2(const int n_threads, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[BLOCK_SIZE], __dma_aligned float cache2[BLOCK_SIZE], const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{	
	//Constraint:
	//Block must fit in WRAM or multiple Blocks must fit in WRAM
	//Block rows can't have more than 512 elements

	int max_threads = m1_rows * m2_columns / 2;
	float t_output[max_threads];
	int n_elem = m1_rows * m2_columns / n_threads;

	if(n_threads > max_threads)
	{
		n_elem = m1_rows * m2_columns / max_threads;
	}
	//printf("n_elem value in kdot is: %d\n", n_elem);
 
	//int max_read_elem = min(512, BLOCK_ROWS * BLOCK_COLUMNS);
	//int max_read_elem2 = min(512, BLOCK_ROWS * BLOCK_COLUMNS);
	//int iter1 = (m1_rows * m1_columns / max_read_elem);
	//int iter2 = (m1_columns * m2_columns / max_read_elem);
	for(int b_i = 0; b_i < 1; b_i++)
	{
		for(int b_j = 0; b_j < 1; b_j++)
		{
		for(int l = 0; l < BLOCK_ROWS ; l++)
		{
			mram_read(&m1[l*m1_columns + b_i*BLOCK_COLS+b_j*m1_columns*BLOCK_COLS], &cache1[l*BLOCK_COLS], sizeof(float)*BLOCK_COLS);
			mram_read(&m2[l*m2_columns + b_i*BLOCK_COLS+b_j*m2_columns*BLOCK_COLS], &cache2[l*BLOCK_COLS], sizeof(float)*BLOCK_COLS);
		}
 

 		/*if(me() == 0 && b_i == 1 && b_j == 1)
		{
			printf("I'm block %d\n", b_i);
			for(int j = 0; j < BLOCK_ROWS*BLOCK_COLS; j++)
			{
				printf("Value in column %d is %.10f\n",j, cache1[j]);
				//printf("%d\n",m1_columns);
			}
		}*/
 
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
        			t_output[me()] += cache1[r * m1_columns + k] * cache2[k * m2_columns + c];
	       

	    			//mram_read(&m1[r * m1_columns + k], &cachea[me()][0+ (k%2)], sizeof(float)*2);
	   				//mram_read(&m2[k * m2_columns + c], &cacheb[me()][0+ (k%2)], sizeof(float)*2);
	        		//t_output[me()] += cachea[me()][0+ (k%2)] * cacheb[me()][0+ (k%2)];
	        		//printf("I'm thread %d and Value in column %d is %.10f\n",me(),k, cache1[r * m1_columns + k]);
	        		//printf("cachea I'm thread %d and Value in column %d is %.10f\n",me(),k, cachea[me()][0]);

	    		}
	    		output[(i/BLOCK_COLS*m2_columns)+b_j*BLOCK_ROWS+b_i*BLOCK_COLS+i-(BLOCK_COLS*(i/BLOCK_COLS))] = t_output[me()];
	    		//barrier_wait(barrier_address);
	    		//printf("Stat is: %u. Value in column %d is %.10f\n",check_stack(),i, t_output[me()]);
			}
		}
		/*barrier_wait(barrier_address);
		if(me() == 0)
		{
			for(int j = 0; j < m1_rows*m2_columns; j++)
			{
				printf("Value in column %d is %.10f\n",j, output[j]);
				//printf("%d\n",m1_columns);
			}
		}
		barrier_wait(barrier_address);*/
		}
	}
	return output;
}

__mram_ptr float* kDot_MRAM(const int n_threads, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{	

	int max_threads = m1_rows * m2_columns / 2;
	float t_output[max_threads];
	int n_elem = m1_rows * m2_columns / n_threads;

	if(n_threads > max_threads)
	{
		n_elem = m1_rows * m2_columns / max_threads;
	}
	//printf("n_elem value in kdot is: %d\n", n_elem);
 


 
 
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
        	t_output[me()] += m1[r * m1_columns + k] * m2[k * m2_columns + c];
	       

	    	//mram_read(&m1[r * m1_columns + k], &cachea[me()][0+ (k%2)], sizeof(float)*2);
	   		//mram_read(&m2[k * m2_columns + c], &cacheb[me()][0+ (k%2)], sizeof(float)*2);
	        //t_output[me()] += cachea[me()][0+ (k%2)] * cacheb[me()][0+ (k%2)];
	        //printf(" m1 I'm thread %d and Value in column %d is %.10f\n",me(),k, m1[r * m1_columns + k]);
	        //printf("cachea I'm thread %d and Value in column %d is %.10f\n",me(),k, cachea[me()][0]);

	    }
	    output[i] = t_output[me()];
	    //barrier_wait(barrier_address);
	    //printf("Stat is: %u. Value in column %d is %.10f\n",check_stack(),i, output[i]);
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