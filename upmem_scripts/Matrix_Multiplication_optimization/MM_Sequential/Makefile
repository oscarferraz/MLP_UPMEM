CC=cc

CFLAGS = -Wall -Wextra -std=c99

SRC=mm_sequential.c
OBJ=$(SRC:.c=.o)

EXEC=mm_sequential

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(EXEC)

%.o: %.c
	$(CC) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)

