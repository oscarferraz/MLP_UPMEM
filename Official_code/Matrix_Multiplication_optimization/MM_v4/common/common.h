#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdint.h>


#define M1_ROWS 8
#define M1_COLS 8

#define M2_ROWS 8
#define M2_COLS 8

#define T float
#define T_IS_FLOAT 1
#define T_IS_INT 0 // only works for signed int
#define T_IS_CHAR 0

#define WORKLOAD_CONFIG 1 // How many output elements does one DPU executes?
#define FITS_IN_WRAM 1  // Do all buffers fit in WRAM?
#define WRAM_CONFIG 1   // If not all buffers fit in WRAM, which matrix goes into WRAM: 0 - all in MRAM, 1 - M1 in WRAM, 2 - M2 in WRAM?


#define MAX_DPUS 64
#define NR_DPUS 64

#endif /* __COMMON_H__ */



