#The following variables must be defined prior to including this
#makefile fragment
#
#GRAPH_URL:  the url path to the file


WGET := wget
TAR  := tar
GZIP := gzip
MATRIX2SNAP := ../matrix2snap.py

GRAPH_FILE := $(notdir $(GRAPH_URL))

all: setup

fetch: $(GRAPH_FILE)

$(GRAPH_FILE):
	$(WGET) -N $(GRAPH_URL)

