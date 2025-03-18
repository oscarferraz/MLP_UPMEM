#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdint.h>

#define GEMM 1
#define NR_DPUS 64
#define MULTIPLE 3

#if GEMM == 1
    #define TRAINING_SIZE (NR_DPUS*MULTIPLE)
#else
    #define TRAINING_SIZE NR_DPUS
#endif

#define TRAINING_DIM 112

#define OUTPUT_SIZE 1

#define y_dim (TRAINING_SIZE*OUTPUT_SIZE)

#define L1_SIZE 96
#define L2_SIZE 64



#define T float

#define IS_FLOAT 1
#define IS_INT 0


#endif /* __COMMON_H__ */
