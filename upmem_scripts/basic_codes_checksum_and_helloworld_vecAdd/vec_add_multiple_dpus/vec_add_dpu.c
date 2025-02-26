#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization

#define NR_ELEM_PER_TASKLETS (NR_ELEM_PER_DPU/NR_TASKLETS)
#define CACHE_SIZE 32
__mram uint32_t buffer1[NR_ELEM_PER_DPU];// the buffer will be stored in WRAM if i dont specify it
__mram uint32_t buffer2[NR_ELEM_PER_DPU];// the buffer will be stored in WRAM if i dont specify it
//buffer in mram with data
__dma_aligned uint32_t cache1[NR_TASKLETS][CACHE_SIZE]; // cache in WRAM to perform transfer
__dma_aligned uint32_t cache2[NR_TASKLETS][CACHE_SIZE]; // cache in WRAM to perform transfer

__host uint32_t sum[NR_ELEM_PER_DPU];

int main()
{
	//me() gets the id of the thread that is running
	if(me() == 0)
	{
		perfcounter_config(COUNT_CYCLES, true);
	}
	barrier_wait(&my_barrier);

	for(int i = me()*NR_ELEM_PER_TASKLETS; i < (me()+1)*NR_ELEM_PER_TASKLETS; i+=CACHE_SIZE)
	{
		//entre a thread atual e a proxima vou iterar em chunks de 32
		//vou ler do buffer em mram para a cache em WRAM cada 32 bytes
		mram_read(&buffer1[i], &cache1[me()][0], sizeof(uint32_t)*CACHE_SIZE);
		mram_read(&buffer2[i], &cache2[me()][0], sizeof(uint32_t)*CACHE_SIZE);
		sum[i/32] = vec_add(&cache1[me()][0], &cache2[me()][0], CACHE_SIZE);
		//printf("%x\n", sum[i/32]);
	}
	barrier_wait(&my_barrier);
	if(me() == 0)
	{
		perfcounter_t end_time = perfcounter_get();
		printf ("Number of cycles used: %lu, number of cycles per thread %f\n", end_time, (float)end_time/NR_ELEM_PER_DPU);
	}
	return 0;
}