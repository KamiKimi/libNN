uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not')
uname_M := $(shell sh -c 'uname -m 2>/dev/null || echo not')

OPTIMIZATION?=-O2
STD=-std=c11 -pedantic
WARN=-Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wmissing-declarations -Wredundant-decls -Wshadow
OPT=$(OPTIMIZATION)

ifneq (,$(findstring gcc, $(CC)))
  WARN+=-Wlogical-op
endif

FLAGS=$(STD) $(WARN) $(OPT)

DEBUG=-g -ggdb

PREFIX?=/usr/local
INSTALL_LIB=$(PREFIX)/lib
INSTALL_INC=$(PREFIX)/include
INSTALL=install

ifeq ($(uname_S),SunOS)
	# SunOS
        ifneq ($(@@),32bit)
		FLAGS+= -m64
	endif
	DEBUG=-g
	INSTALL=cp -pf
endif

NNCC=$(CC) $(FLAGS) $(DEBUG)

OBJECTS=activation.o iter.o model.o predict.o train.o
INCLUDES=activation.h iter.h model.h predict.h train.h
ARCHIVE=libNN.a

all: $(ARCHIVE)

$(ARCHIVE): $(OBJECTS)
	ar cr $(ARCHIVE) $(OBJECTS)

%.o: %.c
	$(NNCC) -c $<

.PHONY: all


clean:
	rm -rf $(OBJECTS)

.PHONY: clean


install: all NN.h
	@mkdir -p $(INSTALL_LIB)
	$(INSTALL) $(ARCHIVE) $(INSTALL_LIB)
	@mkdir -p $(INSTALL_INC)
	$(INSTALL) NN.h $(INSTALL_INC)
	@mkdir -p $(INSTALL_INC)/NN
	$(INSTALL) $(INCLUDES) $(INSTALL_INC)/NN

uninstall:
	rm -f $(INSTALL_LIB)/$(ARCHIVE)
	rm -f $(INSTALL_INC)/NN/{$(shell sh -c "echo $(INCLUDES) | tr ' ' ','")}
	rm -f $(INSTALL_INC)/NN.h