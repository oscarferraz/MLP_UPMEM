#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdint.h>

static inline uint32_t vec_add(uint32_t *v1_data, uint32_t *v2_data, uint32_t nr_elem)
{
    uint32_t sum[nr_elem];
    for(uint32_t i = 0; i < nr_elem; i++)
    {
        sum[i] = v1_data[i] + v2_data[i];
        //printf("%x\n", sum[i]);
    }
    return *sum;
}

#define NR_ELEM_PER_DPU (4*1024)

#endif /* __COMMON_H__ */
