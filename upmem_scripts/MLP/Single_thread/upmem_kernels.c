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

#define EXP_A 1512775.0
#define EXP_C 68243.0 /*see text for choice of values */
#define EXP(y) (_eco.n.i=EXP_A*(y)+(1072693248.0-EXP_C), _eco.d)




#include "fdlibm.h"

#ifdef __STDC__
static const double
#else
static double
#endif
one	= 1.0,
halF[2]	= {0.5,-0.5,},
huge	= 1.0e+300,
twom1000= 9.33263618503218878990e-302,     /* 2**-1000=0x01700000,0*/
o_threshold=  7.09782712893383973096e+02,  /* 0x40862E42, 0xFEFA39EF */
u_threshold= -7.45133219101941108420e+02,  /* 0xc0874910, 0xD52D3051 */
ln2HI[2]   ={ 6.93147180369123816490e-01,  /* 0x3fe62e42, 0xfee00000 */
	     -6.93147180369123816490e-01,},/* 0xbfe62e42, 0xfee00000 */
ln2LO[2]   ={ 1.90821492927058770002e-10,  /* 0x3dea39ef, 0x35793c76 */
	     -1.90821492927058770002e-10,},/* 0xbdea39ef, 0x35793c76 */
invln2 =  1.44269504088896338700e+00, /* 0x3ff71547, 0x652b82fe */
P1   =  1.66666666666666019037e-01, /* 0x3FC55555, 0x5555553E */
P2   = -2.77777777770155933842e-03, /* 0xBF66C16C, 0x16BEBD93 */
P3   =  6.61375632143793436117e-05, /* 0x3F11566A, 0xAF25DE2C */
P4   = -1.65339022054652515390e-06, /* 0xBEBBBD41, 0xC5D26BF1 */
P5   =  4.13813679705723846039e-08; /* 0x3E663769, 0x72BEA4D0 */


#ifdef __STDC__
	double __ieee754_exp(double x)	/* default IEEE double exp */
#else
	double __ieee754_exp(x)	/* default IEEE double exp */
	double x;
#endif
{
	double y,hi,lo,c,t;
	int k,xsb;
	unsigned hx;

	hx  = __HI(x);	/* high word of x */
	xsb = (hx>>31)&1;		/* sign bit of x */
	hx &= 0x7fffffff;		/* high word of |x| */

    /* filter out non-finite argument */
	if(hx >= 0x40862E42) {			/* if |x|>=709.78... */
            if(hx>=0x7ff00000) {
		if(((hx&0xfffff)|__LO(x))!=0) 
		     return x+x; 		/* NaN */
		else return (xsb==0)? x:0.0;	/* exp(+-inf)={inf,0} */
	    }
	    if(x > o_threshold) return huge*huge; /* overflow */
	    if(x < u_threshold) return twom1000*twom1000; /* underflow */
	}

    /* argument reduction */
	if(hx > 0x3fd62e42) {		/* if  |x| > 0.5 ln2 */ 
	    if(hx < 0x3FF0A2B2) {	/* and |x| < 1.5 ln2 */
		hi = x-ln2HI[xsb]; lo=ln2LO[xsb]; k = 1-xsb-xsb;
	    } else {
		k  = (int)(invln2*x+halF[xsb]);
		t  = k;
		hi = x - t*ln2HI[0];	/* t*ln2HI is exact here */
		lo = t*ln2LO[0];
	    }
	    x  = hi - lo;
	} 
	else if(hx < 0x3e300000)  {	/* when |x|<2**-28 */
	    if(huge+x>one) return one+x;/* trigger inexact */
	}
	else k = 0;

    /* x is now in primary range */
	t  = x*x;
	c  = x - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))));
	if(k==0) 	return one-((x*c)/(c-2.0)-x); 
	else 		y = one-((lo-(x*c)/(2.0-c))-hi);
	if(k >= -1021) {
	    __HI(y) += (k<<20);	/* add k to y's exponent */
	    return y;
	} else {
	    __HI(y) += ((k+1000)<<20);/* add k to y's exponent */
	    return y*twom1000;
	}
}










#define NR_ELEM_PER_TASKLETS (TRAINING_SIZE*TRAINING_DIM/NR_TASKLETS)
#define CACHE_SIZE 64


float exponential(float x)
{
    float sum = 1.0f; // initialize sum of series
 	int n = 100;
    for (int i = n - 1; i > 0; --i )
        sum = 1 + x * sum / i;
 
    return sum;
}


__mram_ptr float* kMartixByMatrixElementwise(const int n_threads, const int width, const int height, const int cache_size, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[NR_TASKLETS][CACHE_SIZE], __dma_aligned float cache2[NR_TASKLETS][CACHE_SIZE], struct barrier_t * barrier_address)
{


	int n_elem = width * height/n_threads;
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		mram_read(&m1[i], &cache1[me()][i], sizeof(float)*cache_size);
		mram_read(&m2[i], &cache2[me()][i], sizeof(float)*cache_size);
		output[i] = cache1[me()][i] * cache2[me()][i];
		//printf("Value %d is: %.10f\n", i, output[i]);
	}
	barrier_wait(barrier_address);
	return output;

}

__mram_ptr float* kMartixSubstractMatrix(const int n_threads, const int width, const int height, const int cache_size, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[NR_TASKLETS][CACHE_SIZE], __dma_aligned float cache2[NR_TASKLETS][CACHE_SIZE], struct barrier_t * barrier_address)
{
	int n_elem = width * height/n_threads;
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		mram_read(&m1[i], &cache1[me()][i], sizeof(float)*cache_size);
		mram_read(&m2[i], &cache2[me()][i], sizeof(float)*cache_size);
		output[i] = cache1[me()][i] - cache2[me()][i];
		//printf("%f\n", output[i]);
	}
	barrier_wait(barrier_address);
	return output;

}

__mram_ptr float* kSigmoid(const int n_threads, const int width, const int height, const int cache_size, __mram_ptr float const *m1, __mram_ptr float *output, __dma_aligned float cache1[NR_TASKLETS][CACHE_SIZE], struct barrier_t * barrier_address)
{
    
    int n_elem = width * height/n_threads;
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		mram_read(&m1[i], &cache1[me()][i], sizeof(float)*cache_size);
		//printf("m1 elem %d is: %.10f\n",i, m1[i]);
		output[i] = 1.0 / (1.0 + exponential(-cache1[me()][i]));
		//output[i] = exp(-cache1[me()][i]);
		//printf("%.10f\n", output[i]);
		//printf("sig elem %d is: %.10f\n",i, output[i]);
	}
	barrier_wait(barrier_address);
	return output;
}

__mram_ptr float* kSigmoid_d(const int n_threads, const int width, const int height, const int cache_size, __mram_ptr float const *m1, __mram_ptr float *output, __dma_aligned float cache1[NR_TASKLETS][CACHE_SIZE], struct barrier_t * barrier_address)
{
	int n_elem = width * height/n_threads;
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		mram_read(&m1[i], &cache1[me()][i], sizeof(float)*cache_size);
		output[i] = cache1[me()][i] * (1 - cache1[me()][i]);
		//printf("Value %d, is: %f\n", i, output[i]);
	}
	barrier_wait(barrier_address);
	return output;
}

__mram_ptr float* kDot(const int n_threads, const int cache_size, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[NR_TASKLETS][CACHE_SIZE], __dma_aligned float cache2[NR_TASKLETS][CACHE_SIZE], const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{
	
	int n_elem = m1_rows*m2_columns / n_threads;
	//printf("Nr of elems: %d\n", n_elem);

	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		//printf("iter nr: %d\n",i);
	    int r = (int)(i / m2_columns);
	    int c = i % m2_columns;
	    float t_output = 0.f;

	    for( int k = 0; k < m1_columns; ++k )
	    {
	    	mram_read(&m1[r * m1_columns + k], &cache1[me()][r * m1_columns + k], sizeof(float)*cache_size);
	    	mram_read(&m2[k * m2_columns + c], &cache2[me()][k * m2_columns + c], sizeof(float)*cache_size);
	        t_output += cache1[me()][r * m1_columns + k] * cache2[me()][k * m2_columns + c];
	        //printf("Value in column %d is %f\n",k * m2_columns + c, m2[k * m2_columns + c]);
	    }

	    output[i] = t_output;
	    printf("Value in column %d is %.10f\n",i, output[i]);
	}
	
	barrier_wait(barrier_address);
	return output;
}

__mram_ptr float* kDot_m1_m2T(const int n_threads, const int cache_size, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[NR_TASKLETS][CACHE_SIZE], __dma_aligned float cache2[NR_TASKLETS][CACHE_SIZE], const int m1_rows, const int m1_columns, const int m2_rows, struct barrier_t * barrier_address)
{

	int n_elem = m1_rows*m2_rows / n_threads;
	//printf("Nr of elems: %d\n", n_elem);
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
		//printf("iter nr: %d\n",i);
	    int r = (int)i / m2_rows;
	    int c = i % m2_rows;
	    float t_output = 0.0;
	    int id_T;

	    for( int k = 0; k < m1_columns; ++k )
	    {	
	    	id_T = c * m1_columns + k;
	    	mram_read(&m1[r * m1_columns + k], &cache1[me()][r * m1_columns + k], sizeof(float)*cache_size);
	    	mram_read(&m2[id_T], &cache2[me()][id_T], sizeof(float)*cache_size);
	        t_output += cache1[me()][r * m1_columns + k] * cache2[me()][id_T];
	        //printf("Value in column %d is %f\n",k, t_output);
	    }

	    output[i] = t_output;
	}
	
	barrier_wait(barrier_address);
	return output;
}

__mram_ptr float* kDot_m1T_m2(const int n_threads, const int cache_size, __mram_ptr float const *m1, __mram_ptr float const *m2, __mram_ptr float *output, __dma_aligned float cache1[NR_TASKLETS][CACHE_SIZE], __dma_aligned float cache2[NR_TASKLETS][CACHE_SIZE], const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{
	int n_elem = m1_columns*m2_columns / n_threads;
	for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
	{
	    int r = (int)i / m2_columns;
	    int c = i % m2_columns;
	    float t_output = 0.0;
	    int id_T;

	    for( int k = 0; k < m1_rows; ++k )
	    {
	    	id_T = k * m1_columns + r;
	    	mram_read(&m1[id_T], &cache1[me()][id_T], sizeof(float)*cache_size);
	    	mram_read(&m2[k * m2_columns + c], &cache2[me()][k * m2_columns + c], sizeof(float)*cache_size);
	        t_output += cache1[me()][id_T] * cache2[me()][k*m2_columns + c];
	        //printf("Value %d: %.10f\n", k * m2_columns + c, m2[k * m2_columns + c]);
	        //printf("Value %d: %.10f\n", id_T, m1[id_T]);
	        //printf("ROWS: %d\n", m2_columns);
	        //printf("Value in column %d is %f\n",k, t_output);
	    }

	    output[i] += t_output;
	    printf("Value %d: %.10f\n", i, output[i]);
	}
	
	barrier_wait(barrier_address);
	return output;
}


void kFit(const int n_threads, const int cache_size,
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
		__mram_ptr float *buffer)
{
	for (unsigned i = 0; i < 1; ++i)
	{
		//activation of FCL with no bias x2
		printf("First Sigmoid: \n");
        kSigmoid(n_threads, l1_w, X_h, cache_size, kDot(n_threads, cache_size, X, W0, l1, cache1, cache2, X_h, X_w, l1_w, barrier_address), l1, cache1, barrier_address);
        printf("Second Sigmoid: \n");
        kSigmoid(n_threads, y_w, X_h, cache_size, kDot(n_threads, cache_size, l1, W1, pred, cache1, cache2, X_h, l1_w, y_w, barrier_address), pred, cache1, barrier_address);
        //backpropagation: gradient of y - x
		printf("Backprop: \n");
		kMartixByMatrixElementwise(n_threads, y_w, X_h, cache_size, kMartixSubstractMatrix(n_threads,y_w, X_h, cache_size, y, pred, pred_d, cache1, cache2, barrier_address), kSigmoid_d(n_threads, y_w, X_h, cache_size, pred, buffer, cache1, barrier_address), pred_d, cache1, cache2, barrier_address);
      	printf("\n");
      	kMartixByMatrixElementwise(n_threads, l1_w, X_h, cache_size, kDot_m1_m2T(n_threads, cache_size, pred_d, W1, l_1_d, cache1, cache2, X_h, y_w, l1_w, barrier_address), kSigmoid_d(n_threads, l1_w, X_h, cache_size, l1, buffer, cache1, barrier_address), l_1_d, cache1, cache2, barrier_address);
        //update weights
        printf("Update weights 1\n");
        kDot_m1T_m2(n_threads, cache_size, l1, pred_d, W1, cache1, cache2, X_h, l1_w, y_w, barrier_address);
        printf("Update weights 2\n");
        kDot_m1T_m2(n_threads, cache_size, X, l_1_d, W0, cache1, cache2, X_h, X_w, l1_w, barrier_address);
    	//printf("iter: %d\n", i);
    	for (int i = 0; i < X_w; i++)
    	{
			printf("Prediction[%i] : %f True Value[%i] : %f Error[%i] : %f\n", i, pred[i], i, y[i], i, pred[i] - y[i]);
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





