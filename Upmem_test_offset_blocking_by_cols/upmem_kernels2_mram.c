#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

//#include "math.h"

#define NR_ELEM_PER_TASKLETS (TRAINING_SIZE*TRAINING_DIM/NR_TASKLETS)
#define CACHE_SIZE 1280

static union
{
    double d;
    struct {
      int j,i;
    } n;
 } _eco;

#define EXP_A 1512775
#define EXP_C 68243 /*see text for choice of values */
#define exponential(y) (_eco.n.i=EXP_A*(y)+(1072693248-EXP_C), _eco.d)

/*float exponential(float x)
{
    float sum = 1.0f; // initialize sum of series
 	int n = 100;
    for (int i = n - 1; i > 0; --i )
        sum = 1 + x * sum / i;
 
    return sum;
}*/

int Ceil(float a)
{
    return(a > (int) a) ? (a+1) : (a);
}

int min(int a, int b)
{
	return(a > b) ? b : a;
}

__mram_ptr float* kMartixByMatrixElementwise(const int n_threads, const int width, const int height, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], __dma_aligned float cache2[CACHE_SIZE], struct barrier_t * barrier_address)
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
		output[i] = m1[i] * m2[i];
		//printf("Value of element wise %d is: %.10f\n", i, output[i]);
	}
	}
	barrier_wait(barrier_address);
	return output;

}

__mram_ptr float* kMartixSubstractMatrix(const int n_threads, const int width, const int height, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], __dma_aligned float cache2[CACHE_SIZE], struct barrier_t * barrier_address)
{
	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
	}
	barrier_wait(barrier_address);
	//printf("n_elem in kms value is: %d\n", n_elem);
 
	if(me() < max_threads)
	{
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		output[i] = m1[i] - m2[i];
		//printf("Value %d is: %.10f\n", i, output[i]);
	}
	}
	barrier_wait(barrier_address);
	return output;

}

__mram_ptr float* kSigmoid(const int n_threads, const int width, const int height, __mram_ptr float const *m1, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], struct barrier_t * barrier_address)
{
    
	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;

	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
		//printf("width height: %d\n", n_elem);

	}
	//n_elem = 2;
	//printf("gdgdfgd: %d\n", n_elem);
	
  if(me() < max_threads)
	{
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		//printf("m1 elem %d is: %.10f\n",i, m1[i]);
		output[i] = 1.0 / (1.0 + exponential((int)-m1[i]));
		//output[i] = exp(-cache1[me()][i]);
		//printf("%.10f\n", output[i]);
		//printf("sig elem %d is: %.10f\n",i, m1[i]);

	}
	}
	barrier_wait(barrier_address);


	return output;
}

__mram_ptr float* kSigmoid_inference(const int n_threads, const int width, const int height, __mram_ptr float const *m1, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], struct barrier_t * barrier_address)
{
    
	int max_threads = width * height / 2;
	int n_elem = width * height / n_threads;

	if(max_threads == 0)
		max_threads = 1;

	if(n_threads > max_threads)
	{
		n_elem = width * height / max_threads;
		//printf("width height: %d\n", n_elem);

	}
	//n_elem = 2;
	//printf("gdgdfgd: %d\n", n_elem);
	
  if(me() < max_threads)
	{
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		//printf("m1 elem %d is: %.10f\n",i, m1[i]);
		output[i] = 1.0 / (1.0 + exponential((int)-m1[i]));
		//output[i] = exp(-cache1[me()][i]);
		//printf("%.10f\n", output[i]);
		//printf("sig elem %d is: %.10f\n",i, output[i]);
	}
	}
	barrier_wait(barrier_address);


	return output;
}

__mram_ptr float* kSigmoid_d(const int n_threads, const int width, const int height, __mram_ptr float const *m1, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], struct barrier_t * barrier_address)
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
		output[i] = m1[i] * (1 - m1[i]);
		//printf("Value %d, is: %f\n", i, output[i]);
	}
	}
	barrier_wait(barrier_address);
	return output;
}

__mram_ptr float* kDot(const int n_threads, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], __dma_aligned float cache2[CACHE_SIZE], const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{	
	int max_threads = m1_rows * m2_columns / 2;
	float t_output[max_threads];
	int n_elem = m1_rows * m2_columns / n_threads;
	
	if(n_threads > max_threads)
	{
		n_elem = m1_rows * m2_columns / max_threads;
	}
 

	int max_read_elem = min(512, m1_rows * m1_columns);
	int max_read_elem2 = min(512, m1_columns * m2_columns);
	int iter1 = (m1_rows * m1_columns / max_read_elem);
	int iter2 = (m1_columns * m2_columns / max_read_elem);

	/*for(int i = 0; i < (m1_rows * m1_columns / max_read_elem) ; i++)
	{
		mram_read(&m1[i*max_read_elem], &cache1[i*max_read_elem], sizeof(float)*max_read_elem);

	}
	mram_read(&m1[iter1 * max_read_elem], &cache1[iter1 * max_read_elem], sizeof(float)*(m1_rows * m1_columns % max_read_elem));


	for(int i = 0; i < (m1_columns * m2_columns / max_read_elem) ; i++)
	{
		mram_read(&m2[i*max_read_elem2], &cache2[i*max_read_elem2], sizeof(float)*max_read_elem2);
	}
	mram_read(&m2[iter2 * max_read_elem2], &cache2[iter2 * max_read_elem2], sizeof(float)*(m1_columns * m2_columns % max_read_elem2));	
	*/
  	/*if(me() == 0)
  	{	
  		for(int i = 0; i < m1_columns*m2_columns; i++)
		{
		printf("Valuedada %d is: %.10f\n", i, m2[i]);
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
	if(me() == 0)
	{
		for(int j = 0; j < m1_rows*m2_columns; j++)
		{
			printf("Value in column %d is %.10f\n",j, output[j]);
			//printf("%d\n",m1_columns);
		}
	}
	barrier_wait(barrier_address);
		
	return output;
}

__mram_ptr float* kDot_inference(const int n_threads, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], __dma_aligned float cache2[CACHE_SIZE], const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{	
	int max_threads = m1_rows * m2_columns / 2;
	float t_output[max_threads];
	int n_elem = m1_rows * m2_columns / n_threads;
	
	if(max_threads == 0)
		max_threads = 1;
	
	if(n_threads > max_threads)
	{
		n_elem = m1_rows * m2_columns / max_threads;
	}
 
	int max_read_elem = min(512, m1_rows * m1_columns);
	int max_read_elem2 = min(512, m1_columns * m2_columns);
	int iter1 = (m1_rows * m1_columns / max_read_elem);
	int iter2 = (m1_columns * m2_columns / max_read_elem);

	/*for(int i = 0; i < (m1_rows * m1_columns / max_read_elem) ; i++)
	{
		mram_read(&m1[i*max_read_elem], &cache1[i*max_read_elem], sizeof(float)*max_read_elem);

	}
	mram_read(&m1[iter1 * max_read_elem], &cache1[iter1 * max_read_elem], sizeof(float)*(m1_rows * m1_columns % max_read_elem));


	for(int i = 0; i < (m1_columns * m2_columns / max_read_elem) ; i++)
	{
		mram_read(&m2[i*max_read_elem2], &cache2[i*max_read_elem2], sizeof(float)*max_read_elem2);
	}
	mram_read(&m2[iter2 * max_read_elem2], &cache2[iter2 * max_read_elem2], sizeof(float)*(m1_columns * m2_columns % max_read_elem2));	
	*/
  
  /*for(int i = 0; i < 122*4; i++)
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

__mram_ptr float* kDot_m1_m2T(const int n_threads, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], __dma_aligned float cache2[CACHE_SIZE], const int m1_rows, const int m1_columns, const int m2_rows, struct barrier_t * barrier_address)
{

	//printf("Nr of elems: %d\n", n_elem);
	int max_threads = m1_rows*m2_rows / 2;
	float t_output[max_threads];
	int n_elem = m1_rows*m2_rows / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = m1_rows*m2_rows / max_threads;
	}
	//printf("n_elem value is: %d\n", n_elem);
	
	int max_read_elem = min(512, m1_rows * m1_columns);
  	int max_read_elem2 = min(512, m2_rows * m1_columns);
	int iter1 = (m1_rows * m1_columns / max_read_elem);
  	int iter2 = (m2_rows * m1_columns / max_read_elem);

	/*for(int i = 0; i < (m1_rows * m1_columns / max_read_elem) ; i++)
	{
		mram_read(&m1[i*max_read_elem], &cache1[i*max_read_elem], sizeof(float)*max_read_elem);

	}
	mram_read(&m1[iter1 * max_read_elem], &cache1[iter1 * max_read_elem], sizeof(float)*(m1_rows * m1_columns % max_read_elem));


	for(int i = 0; i < (m2_rows * m1_columns / max_read_elem) ; i++)
	{
		mram_read(&m2[i*max_read_elem2], &cache2[i*max_read_elem2], sizeof(float)*max_read_elem2);
	}
	mram_read(&m2[iter2 * max_read_elem2], &cache2[iter2 * max_read_elem2], sizeof(float)*(m2_rows * m1_columns % max_read_elem2));	
  */
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
	    	//mram_read(&m1[r * m1_columns + k], &cache1[r * m1_columns + k], sizeof(float)*cache_size);
	    	//mram_read(&m2[id_T], &cache2[id_T], sizeof(float)*cache_size);
	        t_output[me()] += m1[r * m1_columns + k] * m2[id_T];
	        //printf("Value in column %d is %f\n",k, t_output[me()]);
	    }

	    output[i] = t_output[me()];
	    printf("Value %d: %.1f\n", i, output[i]);
	}
	}
	barrier_wait(barrier_address);
	return output;
}

__mram_ptr float* kDot_m1T_m2(const float lr, const int n_threads, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[CACHE_SIZE], __dma_aligned float cache2[CACHE_SIZE], const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{

	int max_threads = m1_columns*m2_columns / 2;
	float t_output[max_threads];
	int n_elem = m1_columns*m2_columns / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = m1_columns*m2_columns / max_threads;
	}
	barrier_wait(barrier_address);
	//printf("n_elem value is: %d\n", n_elem);
 
	int max_read_elem = min(512, m1_rows * m1_columns);
	int max_read_elem2 = min(512, m1_rows * m2_columns);
	int iter1 = (m1_rows * m1_columns / max_read_elem);
	int iter2 = (m1_rows * m2_columns / max_read_elem);

	/*for(int i = 0; i < (m1_rows * m1_columns / max_read_elem) ; i++)
	{
		mram_read(&m1[i*max_read_elem], &cache1[i*max_read_elem], sizeof(float)*max_read_elem);

	}
	mram_read(&m1[iter1 * max_read_elem], &cache1[iter1 * max_read_elem], sizeof(float)*(m1_rows * m1_columns % max_read_elem));


	for(int i = 0; i < (m1_rows * m2_columns / max_read_elem) ; i++)
	{
		mram_read(&m2[i*max_read_elem2], &cache2[i*max_read_elem2], sizeof(float)*max_read_elem2);
	}
	mram_read(&m2[iter2 * max_read_elem2], &cache2[iter2 * max_read_elem2], sizeof(float)*(m1_rows * m2_columns % max_read_elem2));	
 	*/
 
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
	        t_output[me()] += m1[id_T] * m2[k*m2_columns + c];
	        //printf("Value %d: %.10f\n", k * m2_columns + c, m2[k * m2_columns + c]);
	        //printf("Value %d: %.10f\n", id_T, m1[id_T]);
	        //printf("ROWS: %d\n", m2_columns);
	        //printf("Value in column %d is %f\n",k, t_output);
	    }
	    //barrier_wait(barrier_address);
	    output[i] +=  lr * t_output[me()];
      //printf("Value %d: %.10f\n", i, output[i]);
         
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


void kFit(const int n_threads,
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
		__mram_ptr float *buffer,
		const float lr)
{
	for (unsigned i = 0; i < 500; ++i)
	{
		//activation of FCL with no bias x2
		//printf("First Sigmoid: \n");
        kSigmoid(n_threads, l1_w, X_h, kDot(n_threads, X, W0, l1, cache1, cache2, X_h, X_w, l1_w, barrier_address), l1, cache1, barrier_address);
        //printf("Second Sigmoid: \n");
        kSigmoid(n_threads, y_w, X_h, kDot(n_threads, l1, W1, pred, cache1, cache2, X_h, l1_w, y_w, barrier_address), pred, cache1, barrier_address);
        //backpropagation: gradient of y - x
		//printf("Backprop: \n");
		kMartixByMatrixElementwise(n_threads, y_w, X_h, kMartixSubstractMatrix(n_threads,y_w, X_h, y, pred, pred_d, cache1, cache2, barrier_address), kSigmoid_d(n_threads, y_w, X_h, pred, buffer, cache1, barrier_address), pred_d, cache1, cache2, barrier_address);
      	//printf("\n");
      	kMartixByMatrixElementwise(n_threads, l1_w, X_h, kDot_m1_m2T(n_threads, pred_d, W1, l_1_d, cache1, cache2, X_h, y_w, l1_w, barrier_address), kSigmoid_d(n_threads, l1_w, X_h, l1, buffer, cache1, barrier_address), l_1_d, cache1, cache2, barrier_address);
        //update weights
        //printf("Update weights W1\n");
        kDot_m1T_m2(lr, n_threads, l1, pred_d, W1, cache1, cache2, X_h, l1_w, y_w, barrier_address);
        //printf("Update weights W0\n");
        kDot_m1T_m2(lr, n_threads, X, l_1_d, W0, cache1, cache2, X_h, X_w, l1_w, barrier_address);
    	//printf("iter: %d\n", i);
	}

	kSigmoid(n_threads, l1_w, X_h, kDot(n_threads, X, W0, l1, cache1, cache2, X_h, X_w, l1_w, barrier_address), l1, cache1, barrier_address);
    kSigmoid(n_threads, y_w, X_h, kDot(n_threads, l1, W1, pred, cache1, cache2, X_h, l1_w, y_w, barrier_address), pred, cache1, barrier_address);
        	

	//barrier_wait(barrier_address);
    	if(me() == 0)
    	{ 
    		for (int i = 0; i < X_h; i++)
    		{
				printf("KFIT Prediction[%i] : %.10f True Value[%i] : %.10f Error[%i] : %f\n", i, pred[i], i, y[i], i, pred[i] - y[i]);
			}
		}
		barrier_wait(barrier_address);  
}

void kTest(const int n_threads,
		__dma_aligned float cache1[CACHE_SIZE], 
		__dma_aligned float cache2[CACHE_SIZE], 
		struct barrier_t *barrier_address,
		__mram_ptr float const *X, const int X_w,  const int X_h,
		 const int y_w,
		__mram_ptr float *l1, const int l1_w,
		__mram_ptr float *pred,
		__mram_ptr float *W0,
		__mram_ptr float *W1)
{

	printf("\nKtest began");
    kSigmoid_inference(n_threads, l1_w, X_h, kDot_inference(n_threads, X, W0, l1, cache1, cache2, X_h, X_w, l1_w, barrier_address), l1, cache1, barrier_address);
    kSigmoid_inference(n_threads, y_w, X_h, kDot_inference(n_threads, l1, W1, pred, cache1, cache2, X_h, l1_w, y_w, barrier_address), pred, cache1, barrier_address);
    //kDot(n_threads, l1, W1, pred, cache1, cache2, X_h, l1_w, y_w, barrier_address);
    if(me() == 0)
    {
    	for (int i = 0; i < X_h; i++)
    	{
    		for(int j = 0; j < X_w; j++)
    		{
    			//printf("matrix = %.10f\n",X[i*X_w+j]);
    		}
			//printf("KTEST Prediction[%i] : %.10f\n", i, pred[i]);
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





