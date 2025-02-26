CC = gcc
CFLAGS = -Wall -g

SRCS = vgg_fcl_seq.c
OBJS = $(SRCS:.c=.o)

TARGET = vgg_fcl_seq

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

mlp_sequential.o: vgg_fcl_seq.c
	$(CC) $(CFLAGS) -c vgg_fcl_seq.c

clean:
	$(RM) $(OBJS) $(TARGET)
