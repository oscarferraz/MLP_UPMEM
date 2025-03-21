#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdint.h>

#define NR_DPUS1 32
#define NR_DPUS2 2

#define TRAINING_SIZE 2048
#define TRAINING_DIM 112
#define OUTPUT_SIZE 1

#define y_dim (TRAINING_SIZE*OUTPUT_SIZE)

#define L1_SIZE 96
#define L2_SIZE 64

#define T float

#endif /* __COMMON_H__ */
