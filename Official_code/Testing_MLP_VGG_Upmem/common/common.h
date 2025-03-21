#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdint.h>

#define NR_DPUS1 8
#define NR_DPUS2 8

#define TRAINING_SIZE 16384
#define TRAINING_DIM 16384
#define OUTPUT_SIZE 1

#define y_dim (TRAINING_SIZE*OUTPUT_SIZE)

#define L1_SIZE 4096
#define L2_SIZE 4096

#define T float

#endif /* __COMMON_H__ */




// #define TRAINING_DIM 176
// #define L1_SIZE 64
// #define L2_SIZE 64
