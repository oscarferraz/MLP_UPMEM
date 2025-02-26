HOST_BINARY := vec_add_host
DPU_BINARY := vec_add_dpu

HOST_SOURCES := vec_add_host.c
DPU_SOURCES := vec_add_dpu.c

.PHONY: all clean run

all: $(HOST_BINARY) $(DPU_BINARY)

clean:
	rm -f $(HOST_BINARY) $(DPU_BINARY)

run: all
	./$(HOST_BINARY)

$(HOST_BINARY): $(DPU_BINARY) $(HOST_SOURCES)
	clang -O2 -DDPU_BINARY='"$(realpath ${DPU_BINARY})"' `dpu-pkg-config --cflags --libs dpu` ${HOST_SOURCES} -o $@

$(DPU_BINARY): $(DPU_SOURCES)
	clang --target=dpu-upmem-dpurte -DNR_TASKLET=16 -O2 $(DPU_SOURCES) -o $@
