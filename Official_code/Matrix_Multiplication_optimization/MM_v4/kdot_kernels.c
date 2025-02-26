#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

//#include "math.h"

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



int Ceil(float a)
{
    return(a > (int) a) ? (a+1) : (a);
}

int min(int a, int b)
{
	return(a > b) ? b : a;
}



__mram_ptr float* kDot(const int n_threads, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{	
	int max_threads = m1_rows * m2_columns / 2;
	// T* t_output = (T*) mem_alloc(max_threads * sizeof(T));
	int n_elem = m1_rows * m2_columns / n_threads;
	
	if(n_threads > max_threads)
	{
		n_elem = m1_rows * m2_columns / max_threads;
	}
 

	//int max_read_elem = min(512, m1_rows * m1_columns);
	//int max_read_elem2 = min(512, m1_columns * m2_columns);
	//int iter1 = (m1_rows * m1_columns / max_read_elem);
	//int iter2 = (m1_columns * m2_columns / max_read_elem);

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
  		for(int i = 0; i < m1_rows; i++)
		{
			for(int j = 0; j < m1_columns; j++)
			{
				printf("%.1f ", m1[i*m1_columns+j]);
			}
			printf("\n");
		}
	}*/
  
 
 	barrier_wait(barrier_address);
	if(me() < max_threads)
	{
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
			//printf("m2 cols: %d\n",m2_columns);
		    int r = (int)i / m2_columns;
		    int c = i % m2_columns;
		    // t_output[me()] = 0.0;
		    barrier_wait(barrier_address);
		    for( int k = 0; k < m1_columns; ++k )
		    {
	        	// t_output[me()] = t_output[me()] + (m1[r * m1_columns + k] * m2[k * m2_columns + c]);
	        	//printf("%f\n",t_output[me()]);
		       	//printf("A = %.1f B = %.1f t_output = %.1f\n", m1[r * m1_columns + k], m2[k * m2_columns + c],t_output[me()]);
	        	//printf("%1f\n", m1[r * m1_columns + k] * m2[k * m2_columns + c]);
		    	//mram_read(&m1[r * m1_columns + k], &cachea[me()][0+ (k%2)], sizeof(float)*2);
		   		//mram_read(&m2[k * m2_columns + c], &cacheb[me()][0+ (k%2)], sizeof(float)*2);
		        //t_output[me()] += cachea[me()][0+ (k%2)] * cacheb[me()][0+ (k%2)];
		        //printf(" m1 I'm thread %d and Value in column %d is %.10f\n",me(),k, m2[k * m2_columns + c]);
		        //printf("cachea I'm thread %d and Value in column %d is %.10f\n",me(),k, cachea[me()][0]);
		        //printf("row is: %d\n", c);
		        //printf("%f\n",m2[k * m2_columns + c]);

		    }
		    // output[i] = t_output[me()];
		    //printf("output is: %f", output[i]);
		    //barrier_wait(barrier_address);
		    //printf("Stat is: %u. Value in column %d is %.10f\n",check_stack(),i, output[i]);
		}
	}
	barrier_wait(barrier_address);
	/*if(me() == 0)
	{
		for(int j = 0; j < m1_rows; j++)
		{
			for(int k = 0; k < m2_columns; k++)
			{
				printf("%.4f ", output[j*m2_columns+k]);
			}
			printf("\n");
		}
	}
	barrier_wait(barrier_address);*/
		
	return output;
}

__mram_ptr float* kDot_inference(const int n_threads, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
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

__dma_aligned float* kDot_m1_m2T( __dma_aligned float const *m1, __dma_aligned float const *m2, __dma_aligned float *output)
{
	
	//each thread must proces at minimum 2 elements (8 bytes)
	//if there are more threads than n_elem*2 then the last threads should not be used.
	//n_elem is the number of elements each thread is going to process
	//printf("Nr of elems: %d\n", n_elem);
	
	#if WORKLOAD_CONFIG > 24
		int n=Ceil((float)(WORKLOAD_CONFIG)/(float)(24));
		for(int i = me()*n; i < me()*n+n; i++){
			if(i>=WORKLOAD_CONFIG){
				return 0;
			}
			output[i]=0;
			for(int j = 0; j < M1_COLS; j++){
				#if WRAM_CONFIG == 0  || WRAM_CONFIG == 1
					output[i]+=m1[j]*m2[i*M2_ROWS+j];
				#endif
				#if WRAM_CONFIG == 2
					output[i]+=m1[i*M1_COLS+j]*m2[j];
				#endif
			}

			// printf("output[%d]=%f\n", i, output[i]);
			// printf("thread=%d\n", me());
		}
	#else
		#if WORKLOAD_CONFIG == 1
			#if M1_COLS > 24
				int n=Ceil((float)(M1_COLS)/(float)(24));
				output[0]=0;
				for(int j = me()*n; j < me()*n+n; j++){
					if(j>=M1_COLS){
						return 0;
					}
					#if WRAM_CONFIG == 0  || WRAM_CONFIG == 1
						output[0]+=m1[j]*m2[i*M1_COLS+j];
					#endif
					#if WRAM_CONFIG == 2
						output[0]+=m1[i*M1_COLS+j]*m2[j];
					#endif
				}
			#else
				if(me()>=M1_COLS){
					return 0;
				}
				output[me()]=0;
				#if WRAM_CONFIG == 0  || WRAM_CONFIG == 1
					output[me()]+=m1[me()]*m2[me()];
					printf("output[%d]=%f, m1[%d]=%f, m2[%d]=%f\n", me(), output[me()], me(), m1[me()], me(), m2[me()]);
					for(){
						
					}
				#endif
				#if WRAM_CONFIG == 2
					output[0]+=m1[i*M1_COLS+j]*m2[j];
				#endif
				

			#endif
		#elif WORKLOAD_CONFIG == 2
		#elif WORKLOAD_CONFIG == 3
		#elif WORKLOAD_CONFIG == 4
			#if M1_COLS > 24
				int n=Ceil((float)(M1_COLS)/(float)(24));
				output[0]=0;
				for(int j = me()*n; j < me()*n+n; j++){
					if(j>=M1_COLS){
						return 0;
					}
					#if WRAM_CONFIG == 0  || WRAM_CONFIG == 1
						output[0]+=m1[j]*m2[i*M1_COLS+j];
					#endif
					#if WRAM_CONFIG == 2
						output[0]+=m1[i*M1_COLS+j]*m2[j];
					#endif
				}
			#else
				output[0]=0;
				if(me()>=M1_COLS){
					return 0;
				}
				#if WRAM_CONFIG == 0  || WRAM_CONFIG == 1
					output[0]+=m1[me()]*m2[me()];
				#endif
				#if WRAM_CONFIG == 2
					output[0]+=m1[i*M1_COLS+j]*m2[j];
				#endif
				

			#endif
		#elif WORKLOAD_CONFIG == 6
		#elif WORKLOAD_CONFIG == 8
		#elif WORKLOAD_CONFIG == 12
		#elif WORKLOAD_CONFIG == 12
		#else
		#endif
		
	#endif



	/* for(int i = 0; i < WORKLOAD_CONFIG; i++){
		output[i]=0;
		for(int j = 0; j < M1_COLS; j++){
			#if WRAM_CONFIG == 0  || WRAM_CONFIG == 1
				output[i]+=m1[j]*m2[i*M2_ROWS+j];
			#endif
			#if WRAM_CONFIG == 2
				output[i]+=m1[i*M1_COLS+j]*m2[j];
			#endif
			
		}

		// printf("output[%d]=%f\n", i, output[i]);
		printf("thread=%d\n", me());
	} */

	return output;
}













