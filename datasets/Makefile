#Makefile to fetch and install graph data for regression
#testing borrowed from Royal Caliber

#Each graph lives in its own directory
SUBDIRS = ak2010 belgium_osm delaunay_n13 delaunay_n21 delaunay_n24 coAuthorsDBLP kron_g500-logn21 soc-LiveJournal1 webbase-1M europe_osm road_usa cit-Patents soc-orkut indochina-2004 hollywood-2009 roadNet-CA

SUBDIRS_IPDPS17 = soc-LiveJournal1 hollywood-2009 soc-orkut soc-sinaweibo soc-twitter-2010 indochina-2004 uk-2002 arabic-2005 uk-2005 webbase-2001 germany_osm asia_osm europe_osm road_central road_usa kron_g500-logn21

SUBDIRS_TOPC = soc-LiveJournal1 hollywood-2009 soc-orkut indochina-2004 road_usa

SUBDIRS_STANDARD = soc-LiveJournal1 hollywood-2009 soc-orkut indochina-2004 road_usa

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

IPDPS17: recurse_ipdps17

recurse_ipdps17:
	for subdir in $(SUBDIRS_IPDPS17); do $(MAKE) -C $$subdir; done

TOPC: recurse_topc

recurse_topc:
	for subdir in $(SUBDIRS_TOPC); do $(MAKE) -C $$subdir; done

STANDARD: recurse_standard

recurse_standard:
	for subdir in $(SUBDIRS_STANDARD); do $(MAKE) -C $$subdir; done

$(GRAPHS):
	$(MAKE) -C $@
