HOST_BINARY := host
KDOT_RELU_BINARY := dpu


HOST_SOURCES := host.c
DPU_SOURCES := dpu.c




.PHONY: all clean run

all: ${HOST_BINARY} ${KDOT_RELU_BINARY} 

clean:
	rm -f ${HOST_BINARY} ${KDOT_RELU_BINARY} 

run: all
	./${HOST_BINARY}

${HOST_BINARY}: ${KDOT_RELU_BINARY} ${HOST_SOURCES}
	clang -O2 -DDPU_BINARY='"$(realpath ${KDOT_RELU_BINARY})" ' `dpu-pkg-config --cflags --libs dpu` ${HOST_SOURCES} -o $@

${KDOT_RELU_BINARY}: ${DPU_SOURCES}
	clang --target=dpu-upmem-dpurte -DNR_TASKLETS=1 -O2 ${DPU_SOURCES} -o $@

