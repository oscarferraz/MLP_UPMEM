#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization

#define BUF_SIZE (4*1024)
#define NR_ELEM_PER_TASKLETS (BUF_SIZE/NR_TASKLETS)
#define CACHE_SIZE 32
__mram uint32_t buffer[BUF_SIZE];// the buffer will be stored in WRAM if i dont specify it
//buffer in mram with data

uint32_t checksums[NR_TASKLETS] = {0}; // variable for the checksum of each thread
__dma_aligned uint32_t cache[NR_TASKLETS][CACHE_SIZE]; // cache in WRAM to perform transfer
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
		mram_read(&buffer[i], &cache[me()][0], sizeof(uint32_t)*CACHE_SIZE);
		checksums[me()] += compute_checksum(&cache[me()][0], CACHE_SIZE);
	}
	barrier_wait(&my_barrier);
	if(me() == 0)
	{
		uint32_t checksum = 0;
		for(int i = 0; i < NR_TASKLETS; i++)
		{
			checksum += checksums[i];
		}
		perfcounter_t end_time = perfcounter_get();
		printf ("checksum: %u, number of cicles used: %lu, number of cicles per thread %f\n",checksum, end_time, (float)end_time/BUF_SIZE);
	}
	return 0;
}