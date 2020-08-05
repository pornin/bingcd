CC = clang
CFLAGS = -W -Wall -O3 -march=native
LD = clang
LDFLAGS =
LIBS =

OBJ = gf25519.o
TESTOBJ = sha3.o test_gf25519.o
TESTLIBS = -lgmp

all: test_gf25519

clean:
	-rm -f $(OBJ) $(TESTOBJ) test_gf25519

test_gf25519: $(OBJ) $(TESTOBJ)
	$(LD) $(LDFLAGS) -o test_gf25519 $(OBJ) $(TESTOBJ) $(LIBS) $(TESTLIBS)

gf25519.o: gf25519.c gf25519.h
	$(CC) $(CFLAGS) -c -o gf25519.o gf25519.c

sha3.o: sha3.c sha3.h
	$(CC) $(CFLAGS) -c -o sha3.o sha3.c

test_gf25519.o: test_gf25519.c gf25519.h sha3.h
	$(CC) $(CFLAGS) -c -o test_gf25519.o test_gf25519.c
