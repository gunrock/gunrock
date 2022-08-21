#common make file fragment for networkrepository.com
#just define GRAPH_NAME prior to including this fragment

GRAPH_ZIP  = $(GRAPH_NAME).zip

setup: $(GRAPH_NAME).mtx

$(GRAPH_NAME).mtx: $(GRAPH_ZIP)
	unzip $(GRAPH_ZIP)
	rm -rf readme.txt

clean:
	rm $(GRAPH_NAME).mtx

realclean: clean
	rm $(GRAPH_ZIP)

