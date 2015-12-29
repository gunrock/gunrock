#Makefile to fetch and install graph data for regression
#testing borrowed from Royal Caliber

#Each graph lives in its own directory
SUBDIRS = ak2010 belgium_osm delaunay_n13 delaunay_n21 delaunay_n24 coAuthorsDBLP kron_g500-logn21 soc-LiveJournal1 webbase-1M europe_osm road_usa cit-Patents soc-orkut indochina-2004 hollywood-2009 roadNet-CA

.PHONY: $(GRAPHS)

#fetches all graphs, extracts and sets up files for tests
all: recurse

#only download the graphs, but do not proceed further
fetch: recurse

#clean everything except the downloaded graphs
clean: recurse

#clean everything including the downloaded graphs
realclean: recurse

#recurse into each subdirectory and try to build the provided targets
recurse:
	for subdir in $(SUBDIRS); do $(MAKE) -C $$subdir $(MAKECMDGOALS); done


$(GRAPHS):
	$(MAKE) -C $@
