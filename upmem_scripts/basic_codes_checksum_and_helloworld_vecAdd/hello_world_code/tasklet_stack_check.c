#include <stdio.h>
#include <defs.h>

int main()
{
    printf("tasklet %u: stack = %u\n", me(), check_stack());
    return 0;
}

//check_stat() returns the remaining available size in the stack.
//me() fetches each tasklet system name

//dpu-upmem-dpurte-clang -DNR_TASKLETS=3 -DSTACK_SIZE_DEFAULT=256 -DSTACK_SIZE_TASKLET_1=2048 -O2 -o tasklet_stack_check tasklet_stack_check.c

//NR_TASKLETS is used to define the number of tasklets.

//STACK_SIZE_DEFAULT is used to define the size of the stack for all the tasklets which stack is not specified.

//STACK_SIZE_TASKLET_<X> is used to define the size of the stack for a specific tasklet.