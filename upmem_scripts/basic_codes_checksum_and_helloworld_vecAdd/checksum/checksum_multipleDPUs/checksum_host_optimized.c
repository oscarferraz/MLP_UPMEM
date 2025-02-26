#include <stdio.h>
#include <stdlib.h>
#include <dpu.h>
#include "common/common.h"

static void init_array(uint32_t *buffer, int nr_elem)
{
	srand(0);
	for(int i = 0; i < nr_elem; i++)
	{
		buffer[i] = rand();
	}
}


static void free_buffers(uint32_t *input_array, uint32_t *output_array)
{
	free(input_array);
	free(output_array);
}

static void alloc_buffers(uint32_t **input_array, uint32_t **output_array, uint32_t nr_dpus)
{
	*input_array = malloc(sizeof(uint32_t) * NR_ELEM_PER_DPU * nr_dpus);
	*output_array = malloc(sizeof(uint32_t) * nr_dpus);

	init_array(*input_array, NR_ELEM_PER_DPU * nr_dpus);
}

static void free_dpus(struct dpu_set_t set)
{}

static void alloc_dpus(struct dpu_set_t *set, uint32_t *nr_dpus)
{
	DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, set));
	DPU_ASSERT(dpu_load(*set, DPU_BINARY, NULL));

	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
}

int main()
{
	struct dpu_set_t set, dpu;
    uint32_t nr_dpus, each_dpu;
    alloc_dpus(&set, &nr_dpus);

    uint32_t *input_array, *output_array;
    alloc_buffers(&input_array, &output_array, nr_dpus);

    printf("DPUs allocated: %u\n", nr_dpus);

	//Copy inputs to DPU
	DPU_FOREACH(set,dpu,each_dpu)
	{
		DPU_ASSERT(dpu_prepare_xfer(dpu, &input_array[each_dpu * NR_ELEM_PER_DPU]));
	}
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "buffer", 0, sizeof(uint32_t) * NR_ELEM_PER_DPU, DPU_XFER_DEFAULT));
	
	//Run DPUS
	DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
	
	//Copy outputs from DPUS
	DPU_FOREACH(set,dpu,each_dpu)
	{
		DPU_ASSERT(dpu_prepare_xfer(dpu, &output_array[each_dpu]));
		//DPU_ASSERT(dpu_log_read(dpu,stdout)); //prints dpu contente
	}
	DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "checksum", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));

	//Verify computation
    uint32_t expected_checksum = compute_checksum(input_array, NR_ELEM_PER_DPU * nr_dpus);
    uint32_t checksum = compute_checksum(output_array, nr_dpus);

    printf("expected: 0x%x - got: 0x%x\n", expected_checksum, checksum);

    if(checksum == expected_checksum)
    {
    	printf("OK!\n");
    }
    else
    {
    	printf("KO...!\n");
    }
    free_buffers(input_array, output_array);
    free_dpus(set);
    return 0;
}

