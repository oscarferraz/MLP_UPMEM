#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include "common/common.h"

static void init_array(uint32_t *buffer, int nr_elem)
{
	srand(0);
	for(int i = 0; i < nr_elem; i++)
	{
		buffer[i] = 1;
	}
}


static void free_buffers(uint32_t *input_array_1, uint32_t *input_array_2, uint32_t *output_array)
{
	free(input_array_1);
	free(input_array_2);
	free(output_array);
}

static void alloc_buffers(uint32_t **input_array_1, uint32_t **input_array_2, uint32_t **output_array, uint32_t nr_dpus)
{
	*input_array_1 = malloc(sizeof(uint32_t) * NR_ELEM_PER_DPU * nr_dpus);
	*input_array_2 = malloc(sizeof(uint32_t) * NR_ELEM_PER_DPU * nr_dpus);
	*output_array = malloc(sizeof(uint32_t) * NR_ELEM_PER_DPU * nr_dpus);

	init_array(*input_array_1, NR_ELEM_PER_DPU * nr_dpus);
	init_array(*input_array_2, NR_ELEM_PER_DPU * nr_dpus);
}

static void free_dpus(struct dpu_set_t set)
{}

static void alloc_dpus(struct dpu_set_t *set, uint32_t *nr_dpus)
{
	DPU_ASSERT(dpu_alloc(1, NULL, set));
	DPU_ASSERT(dpu_load(*set, DPU_BINARY, NULL));

	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
}

int main()
{
	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    alloc_dpus(&set, &nr_dpus);

    uint32_t *input_array_1, *input_array_2, *output_array;
    alloc_buffers(&input_array_1, &input_array_2, &output_array, nr_dpus);

    printf("DPUs allocated: %u\n", nr_dpus);

	//Copy inputs to DPU
	DPU_ASSERT(dpu_copy_to(set, "buffer1", 0, input_array_1, sizeof(uint32_t) * NR_ELEM_PER_DPU));
	DPU_ASSERT(dpu_copy_to(set, "buffer2", 0, input_array_2, sizeof(uint32_t) * NR_ELEM_PER_DPU));

	//Run DPUS
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
	//Copy outputs from DPUS
	DPU_FOREACH(set,dpu)
	{
		DPU_ASSERT(dpu_copy_from(dpu, "sum", 0, output_array, sizeof(uint32_t) * NR_ELEM_PER_DPU));
		DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu content
	}

	//Verify computation
    //uint32_t expected_sum = vec_add(input_array, NR_ELEM_PER_DPU * nr_dpus);
    //uint32_t sum = compute_checksum(output_array, nr_dpus);


    /*if(expected_sum == sum)
    {
    	printf("OK!\n");
    }
    else
    {
    	printf("KO...!\n");
    }*/
    for(int i = 0; i < 100; i++)
    {
    	printf("%x\n", output_array[i]);
    }
    free_buffers(input_array_1, input_array_2, output_array);
    free_dpus(set);
    return 0;
}