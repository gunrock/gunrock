#The following variables must be defined prior to including this
#makefile fragment
#
#GRAPH_URL:  the url path to the file

OSUPPER := $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

ifeq (DARWIN, $(findstring DARWIN, $(OSUPPER)))
    WGET := curl -O
else
    WGET := wget -N
endif

TAR  := tar
GZIP := gzip
MATRIX2SNAP := ../matrix2snap.py

GRAPH_FILE := $(notdir $(GRAPH_URL))

all: setup

fetch: $(GRAPH_FILE)

$(GRAPH_FILE):
	$(WGET) $(GRAPH_URL)

IPDPS17: setup
