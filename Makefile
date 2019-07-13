.SUFFIXES: .c .u
CC= gcc

CFLAGS = -g -Wall -I/usr/local/Cellar/gsl/2.4/include -O3 -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF 
LDFLAGS = -L/usr/local/lib -lgsl -lgslcblas
# LDFLAGS = -L/usr/local/lib -lgsl -lcblas
LOBJECTS= inference.o gsl-wrappers.o ctm.o estimate.o corpus.o params.o
LSOURCE= inference.c gsl-wrappers.c corpus.c estimate.c corpus.c params.c

mac:	$(LOBJECTS)
	$(CC) $(LOBJECTS) -o ctm $(LDFLAGS)

linux:	$(LOBJECTS)
	$(CC) $(LOBJECTS) -o ctm $(LDFLAGS)

debug:	$(LOBJECTS)
	$(CC) $(LOBJECTS) -o ctm $(LDFLAGS)

clean:
	-rm -f *.o 
