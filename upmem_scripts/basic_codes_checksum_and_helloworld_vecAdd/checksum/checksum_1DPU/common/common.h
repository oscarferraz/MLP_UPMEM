#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdint.h>

static inline uint32_t compute_checksum(uint32_t *data, uint32_t nr_elem)
{
    uint32_t checksum = 0;
    for(uint32_t i = 0; i < nr_elem; i++)
    {
        checksum += data[i];
    }
    return checksum;
}

#define NR_ELEM_PER_DPU (4*1024)

#endif /* __COMMON_H__ */
