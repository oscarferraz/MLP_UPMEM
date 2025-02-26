#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>

#define BUF_SIZE 4

__mram uint32_t buffer[BUF_SIZE] = {1,2,3,4}; // the buffer will be stored in WRAM if i dont specify it

int main()
{
	printf("HelloWorld!\n");
	perfcounter_config(COUNT_CYCLES, true);
	uint32_t checksum = 0;
	for(int i = 0; i < 512; i++)
	{
		checksum += buffer[i];
	}
	perfcounter_t end_time = perfcounter_get();
	printf ("checksum: %u, number of cicles used: %lu\n",checksum, end_time);
	return 0;
}

/*Neste caso estamos a acessar a MRAM através de um ciclo for o que não é muito 
eficiente pois é um acesso implicito.
Nos queremos fazer acessos explicitos a MRAM para ser mais eficiente. Para além
disso como toda a dpu esta na DRAM die uma thread não pode executar mais do que
uma instrução a cada 11 ciclos*/

