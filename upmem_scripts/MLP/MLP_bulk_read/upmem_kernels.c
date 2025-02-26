#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

//#include "math.h"

#define NR_ELEM_PER_TASKLETS (TRAINING_SIZE*TRAINING_DIM/NR_TASKLETS)
#define CACHE_SIZE 768

float exponential(float x)
{
    float sum = 1.0f; // initialize sum of series
 	int n = 100;
    for (int i = n - 1; i > 0; --i )
        sum = 1 + x * sum / i;
 
    return sum;
}


__mram_ptr float* kMartixByMatrixElementwise(const int n_threads, const int width, const int height, const int cache_size, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], __dma_aligned float cache2[CACHE_SIZE], struct barrier_t * barrier_address)
{

	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
	}
	if(me() < max_threads)
	{	
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		//mram_read(&m1[i], &cache1[i], sizeof(float)*cache_size);
		//mram_read(&m2[i], &cache2[i], sizeof(float)*cache_size);
		output[i] = m1[i] * m2[i];
		//printf("Value %d is: %.10f\n", i, output[i]);
	}
	}
	barrier_wait(barrier_address);
	return output;

}

__mram_ptr float* kMartixSubstractMatrix(const int n_threads, const int width, const int height, const int cache_size, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], __dma_aligned float cache2[CACHE_SIZE], struct barrier_t * barrier_address)
{
	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
	}
	barrier_wait(barrier_address);
	printf("n_elem in kms value is: %d\n", n_elem);
	if(me() < max_threads)
	{
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		mram_read(&m1[i], &cache1[i], sizeof(float)*cache_size);
		mram_read(&m2[i], &cache2[i], sizeof(float)*cache_size);
		output[i] = cache1[i] - cache2[i];
		//printf("Value %d is: %.10f\n", i, output[i]);
	}
	}
	barrier_wait(barrier_address);
	return output;

}

__mram_ptr float* kSigmoid(const int n_threads, const int width, const int height, const int cache_size, __mram_ptr float const *m1, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], struct barrier_t * barrier_address)
{
    
	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
		//printf("width height: %d\n", n_elem);

	}
	//n_elem = 2;
	printf("gdgdfgd: %d\n", n_elem);
	if(me() < max_threads)
	{
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		mram_read(&m1[i], &cache1[i], sizeof(float)*cache_size);
		//printf("m1 elem %d is: %.10f\n",i, m1[i]);
		output[i] = 1.0 / (1.0 + exponential(-cache1[i]));
		//output[i] = exp(-cache1[me()][i]);
		//printf("%.10f\n", output[i]);
		//printf("sig elem %d is: %.10f\n",i, output[i]);
	}
	}
	barrier_wait(barrier_address);


	return output;
}

__mram_ptr float* kSigmoid_d(const int n_threads, const int width, const int height, const int cache_size, __mram_ptr float const *m1, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], struct barrier_t * barrier_address)
{
	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
	}
	if(me() < max_threads)
	{	
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		mram_read(&m1[i], &cache1[i], sizeof(float)*cache_size);
		output[i] = cache1[i] * (1 - cache1[i]);
		//printf("Value %d, is: %f\n", i, output[i]);
	}
	}
	barrier_wait(barrier_address);
	return output;
}

__mram_ptr float* kDot(const int n_threads, const int cache_size, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], __dma_aligned float cache2[CACHE_SIZE], const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{	

	int max_threads = m1_rows * m2_columns / 2;
	float t_output[max_threads];
	int n_elem = m1_rows * m2_columns / n_threads;

	if(n_threads > max_threads)
	{
		n_elem = m1_rows * m2_columns / max_threads;
	}
	printf("n_elem value in kdot is: %d\n", n_elem);
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
	    	mram_read(&m1[r * m1_columns + k], &cache1[r * m1_columns + k], sizeof(float)*cache_size);
	    	mram_read(&m2[k * m2_columns + c], &cache2[k * m2_columns + c], sizeof(float)*cache_size);
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

__mram_ptr float* kDot_m1_m2T(const int n_threads, const int cache_size, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], __dma_aligned float cache2[CACHE_SIZE], const int m1_rows, const int m1_columns, const int m2_rows, struct barrier_t * barrier_address)
{

	//printf("Nr of elems: %d\n", n_elem);
	int max_threads = m1_rows*m2_rows / 2;
	float t_output[max_threads];
	int n_elem = m1_rows*m2_rows / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = m1_rows*m2_rows / max_threads;
	}
	printf("n_elem value is: %d\n", n_elem);
	if(me() < max_threads)
	{
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		//printf("iter nr: %d\n",i);
	    int r = (int)i / m2_rows;
	    int c = i % m2_rows;
	    t_output[me()] = 0.0;
	    int id_T;

	    for( int k = 0; k < m1_columns; ++k )
	    {	
	    	id_T = c * m1_columns + k;
	    	mram_read(&m1[r * m1_columns + k], &cache1[r * m1_columns + k], sizeof(float)*cache_size);
	    	mram_read(&m2[id_T], &cache2[id_T], sizeof(float)*cache_size);
	        t_output[me()] += cache1[r * m1_columns + k] * cache2[id_T];
	        //printf("Value in column %d is %f\n",k, t_output);
	    }

	    output[i] = t_output[me()];
	    //printf("Value %d: %.10f\n", i, output[i]);
	}
	}
	barrier_wait(barrier_address);
	return output;
}

__mram_ptr float* kDot_m1T_m2(const int n_threads, const int cache_size, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], __dma_aligned float cache2[CACHE_SIZE], const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{

	int max_threads = m1_columns*m2_columns / 2;
	float t_output[max_threads];
	int n_elem = m1_columns*m2_columns / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = m1_columns*m2_columns / max_threads;
	}
	barrier_wait(barrier_address);
	printf("n_elem value is: %d\n", n_elem);
	if(me() < max_threads)
	{
	//printf("Im KDotT and I'm using thread nr: %d. The max is : %d\n", me(), max_threads);	
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
	    int r = (int)i / m2_columns;
	    int c = i % m2_columns;
	    t_output[me()] = 0.0;
	    int id_T;

	    for( int k = 0; k < m1_rows; ++k )
	    {
	    	id_T = k * m1_columns + r;
	    	mram_read(&m1[id_T], &cache1[id_T], sizeof(float)*cache_size);
	    	mram_read(&m2[k * m2_columns + c], &cache2[k * m2_columns + c], sizeof(float)*cache_size);
	        t_output[me()] += cache1[id_T] * cache2[k*m2_columns + c];
	        //printf("Value %d: %.10f\n", k * m2_columns + c, m2[k * m2_columns + c]);
	        //printf("Value %d: %.10f\n", id_T, m1[id_T]);
	        //printf("ROWS: %d\n", m2_columns);
	        //printf("Value in column %d is %f\n",k, t_output);
	    }
	    //barrier_wait(barrier_address);
	    output[i] += t_output[me()];
	}
	}
	barrier_wait(barrier_address);
	/*if(me() == 0)
	{
		for(int i = 0; i < max_threads; i++)
		{
			printf("Value %d: %.10f\n", i, output[i]);
		}
	}
	barrier_wait(barrier_address);*/
	return output;
}


void kFit(const int n_threads, const int cache_size,
		__dma_aligned float cache1[CACHE_SIZE], 
		__dma_aligned float cache2[CACHE_SIZE], 
		struct barrier_t *barrier_address,
		__mram_ptr float const *X, const int X_w,  const int X_h,
		__mram_ptr float const *y, const int y_w,
		__mram_ptr float *l1, const int l1_w,
		__mram_ptr float *l_1_d,
		__mram_ptr float *pred, __mram_ptr float *pred_d,
		__mram_ptr float *W0,
		__mram_ptr float *W1,
		__mram_ptr float *buffer)
{
	for (unsigned i = 0; i < 50; ++i)
	{
		//activation of FCL with no bias x2
		//printf("First Sigmoid: \n");
        kSigmoid(n_threads, l1_w, X_h, cache_size, kDot(n_threads, cache_size, X, W0, l1, cache1, cache2, X_h, X_w, l1_w, barrier_address), l1, cache1, barrier_address);
        printf("Second Sigmoid: \n");
        kSigmoid(n_threads, y_w, X_h, cache_size, kDot(n_threads, cache_size, l1, W1, pred, cache1, cache2, X_h, l1_w, y_w, barrier_address), pred, cache1, barrier_address);
        //backpropagation: gradient of y - x
		//printf("Backprop: \n");
		kMartixByMatrixElementwise(n_threads, y_w, X_h, cache_size, kMartixSubstractMatrix(n_threads,y_w, X_h, cache_size, y, pred, pred_d, cache1, cache2, barrier_address), kSigmoid_d(n_threads, y_w, X_h, cache_size, pred, buffer, cache1, barrier_address), pred_d, cache1, cache2, barrier_address);
      	//printf("\n");
      	kMartixByMatrixElementwise(n_threads, l1_w, X_h, cache_size, kDot_m1_m2T(n_threads, cache_size, pred_d, W1, l_1_d, cache1, cache2, X_h, y_w, l1_w, barrier_address), kSigmoid_d(n_threads, l1_w, X_h, cache_size, l1, buffer, cache1, barrier_address), l_1_d, cache1, cache2, barrier_address);
        //update weights
        printf("Update weights W1\n");
        kDot_m1T_m2(n_threads, cache_size, l1, pred_d, W1, cache1, cache2, X_h, l1_w, y_w, barrier_address);
        printf("Update weights W0\n");
        kDot_m1T_m2(n_threads, cache_size, X, l_1_d, W0, cache1, cache2, X_h, X_w, l1_w, barrier_address);
    	//printf("iter: %d\n", i);
	}
	//barrier_wait(barrier_address);
    	if(me() == 0)
    	{
    		for (int i = 0; i < X_w; i++)
    		{
				printf("Prediction[%i] : %.10f True Value[%i] : %.10f Error[%i] : %f\n", i, pred[i], i, y[i], i, pred[i] - y[i]);
			}
		}  
}

/*void kTest(const int n_elem, const int cache_size,
		__dma_aligned float cache1[NR_TASKLETS][CACHE_SIZE], 
		__dma_aligned float cache2[NR_TASKLETS][CACHE_SIZE], 
		struct barrier_t *barrier_address,
		__mram_ptr float const *X, const int X_w,  const int X_h,
		__mram_ptr float const *y, const int y_w,
		__mram_ptr float *l1, const int l1_w,
		__mram_ptr float *l_1_d,
		__mram_ptr float *pred, __mram_ptr float *pred_d,
		__mram_ptr float *W0,
		__mram_ptr float *W1,
		__mram_ptr float *buffer,
		__mram_ptr float *prediction)
{


	kSigmoid(n_elem, cache_size, kDot(n_elem, cache_size, X, W0, l1, cache1, cache2, X_h, X_w, l1_w, barrier_address), l1, cache1, barrier_address);
	prediction = kSigmoid(n_elem, cache_size, kDot(n_elem, cache_size, l1, W1, pred, cache1, cache2, X_h, l1_w, y_w, barrier_address), pred, cache1, barrier_address);
    barrier_wait(barrier_address);
    //return prediction;    
}*/





