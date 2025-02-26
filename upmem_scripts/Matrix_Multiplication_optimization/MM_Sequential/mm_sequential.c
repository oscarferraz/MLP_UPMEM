#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common/common.h"
#include "common/timer.h"

//Sequential version for Matrix-Multiplication

T* kDot_m1_m2T( T const *m1, T const *m2, T *output, const int m1_rows, const int m1_columns, const int m2_rows)
{

	//second matrix is going to be transposed for multiplication
	int	n_elem = m1_rows*m2_rows;
	//printf("n_elem value is: %d\n", n_elem);
	

	for(int i = 0; i < n_elem; i++)
	{
		  //printf("iter nr: %d\n",i);
	    int r = (int)i / m2_rows;
	    int c = i % m2_rows;
	    int id_T;

	    for( int k = 0; k < m1_columns; ++k )
	    {	
	    	id_T = c * m1_columns + k;
        output[i] += m1[r * m1_columns + k] * m2[id_T];
	    }

	    //printf("Value %d: %.1f\n", i, output[i]);
	}
	
	return output;
}



int main()
{
	//The matrix m1 should be row major
	T* m1 = (T*)malloc(M1_ROWS*M1_COLS*sizeof(T));
	for(int i = 0; i < M1_ROWS*M1_COLS; i++)
	{
		m1[i] = ((float)2);
	}
	
  	//The matrix m2 should be column major
	const long signed int W0_size = M2_ROWS*M1_COLS*sizeof(T);
	T *m2 = (T*)malloc(W0_size);
	for(int i = 0; i < M2_ROWS*M1_COLS; i++)
	{
		m2[i] = ((float)5);
	}

	const long signed int res_size = M1_ROWS*M2_ROWS*sizeof(T);
	T* res = (T*)malloc(res_size);



 	Timer timer;
	start(&timer, 0, 0);

  	//Do multiplication	

  	kDot_m1_m2T( m1, m2, res, M1_ROWS, M1_COLS, M2_ROWS);

  	stop(&timer, 0);
	printf("GEMM Sequential Time (ms): ");
	print(&timer, 0, 1);

	/*printf("\n\n\n\n");
	for(int i = 0; i < M1_ROWS; i++)
	{
		for(int j = 0; j < M2_ROWS; j++)
		{
			printf("%.1f ", res[i*M2_ROWS+j]);
		}
		printf("\n");
	}
	printf("\n\n\n\n");*/

	return 0;
}


