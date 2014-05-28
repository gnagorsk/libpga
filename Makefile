CC=nvcc
ARCH=-arch=sm_20
CFLAGS=-dc -Iinclude $(ARCH)
SOURCES=$(shell echo src/*.cu)
HEADERS=$(shell echo include/*.h)
OBJECTS=$(SOURCES:.cu=.o)
STATIC_LIB=libpga.a

.PHONY: clean test

all: $(STATIC_LIB)

$(STATIC_LIB): $(OBJECTS)
	$(CC) $(ARCH) -lib $(OBJECTS) -o $@

%.o: %.cu
	$(CC) $(CFLAGS) $< -o $@

test:
	$(MAKE) -C test
	test/test
clean:
	rm -f $(STATIC_LIB) $(OBJECTS)
