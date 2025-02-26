HOST_BINARY := mlp_host
DPU_BINARY := mlp_dpu
DPU_BINARY2 := mlp_dpu_other_func

HOST_SOURCES := mlp_host2.c
DPU_SOURCES := mlp_dpu2.c 
DPU_SOURCES_1 := mlp_dpu_other_func.c



.PHONY: all clean run

all: ${HOST_BINARY} ${DPU_BINARY} ${DPU_BINARY2}

clean:
	rm -f ${HOST_BINARY} ${DPU_BINARY} ${DPU_BINARY2}

run: all
	./${HOST_BINARY}

${HOST_BINARY}: ${DPU_BINARY} ${DPU_BINARY2} ${HOST_SOURCES}
	clang -O2 -DDPU_BINARY='"$(realpath ${DPU_BINARY})" "$(realpath ${DPU_BINARY2})"' `dpu-pkg-config --cflags --libs dpu` ${HOST_SOURCES} -o $@

${DPU_BINARY}: ${DPU_SOURCES}
	clang --target=dpu-upmem-dpurte -DNR_TASKLETS=1 -O2 ${DPU_SOURCES} -o $@


${DPU_BINARY2}: ${DPU_SOURCES_1}
	clang --target=dpu-upmem-dpurte -DNR_TASKLETS=1 -O2 ${DPU_SOURCES_1} -o $@
