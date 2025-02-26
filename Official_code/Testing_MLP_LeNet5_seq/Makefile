CC = gcc
CFLAGS = -Wall -g

SRCS = lenet5_fcl_seq.c
OBJS = $(SRCS:.c=.o)

TARGET = lenet5_fcl_seq

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

mlp_sequential.o: lenet5_fcl_seq.c
	$(CC) $(CFLAGS) -c lenet5_fcl_seq.c

clean:
	$(RM) $(OBJS) $(TARGET)
