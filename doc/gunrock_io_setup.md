How to setup and use gunrock/io on a server/machine
================

gunrock/io can be used to generate visual representation of graph engine performance, for exmaple, Gunrock. 
It takes output of a graph algorithm run and can produce visual output in svg, pdf, png, html and md format. 

**Grunrock/io Dependencies**

To use gunrock/io to produce visual output of any graph algorithm, (as of Dec.2016), below are dependencies overview:

* python 2.7.6
* gcc 4.8.4
* nodejs/node
* npm
* pandas 0.19.2
* numpy 1.11.3
* altair 1.2.0
* vega 2.6.3
* vega-lite 1.3.1
* inkscape 0.91


Below are the instructions to to install dependencies, 

**Assume the machine has the following env setup:**

* OS: Ubuntu 14.04 
* Python 2.7.x
* GCC 4.8.x

**Nodejs and Npm**

 First check if node and npm have been installed:

	On command line type:
	node -v
	npm -v
If there is node and npm version output, move on to install altair

Install nodejs and npm on root:

	sudo apt-get install libcairo2-dev libjpeg8-dev libpango1.0-dev libgif-dev build-essential g++
	sudo apt-get install nodejs
	sudo apt-get install npm
	#Create a symbolic link for node, as many Node.js tools use this name to execute.
	sudo ln -s /usr/bin/nodejs /usr/bin/node

**Install altair**

	sudo pip install altair

If no root access, use following command:

	pip install --user altair
	vim ~/.bashrc
	HOME=/home/user
	PYTHONPATH=$HOME/.local/lib/python2.7/site-packages
	source ~/.bashrc

**More altair depencies to save figures**

	npm install -g vega@2.6.3 vega-lite
	#"-g" option install npm/node_modules in /usr/local/bin
	npm -g bin
	#returns directory of installed binary files
	ls [returned directory]
    #check if {vg2png  vg2svg  vl2png  vl2svg  vl2vg} exist in [returned directory]


If no root access, use following command:

	npm install vega@2.6.3 vega-lite
	npm bin
	ls [returned directory]
	#npm install /node_modules in current directory
	#check if {vg2png  vg2svg  vl2png  vl2svg  vl2vg} exist in /bin or /.bin
	#Open .bashrc add: 
	NPM_PACKAGES=/where/node_modules/folder/is/
	PATH=$NPM_PACKAGES/.bin:$PATH
	source ~/.bashrc
	

**More dependencies to save figure as pdf: inkscape**

	sudo add-apt-repository ppa:inkscape.dev/stable
	sudo apt-get update
	sudo apt-get install inkscape

How to use gunrock/io
================

With all the dependencies installed, to use gunrock/io, below is a guide of how to reproduce the performance figures from JSON in gunrock/io:


1.  Parses the engine outputs (in txt format) and generates jsons containing important information regarding the output results using **text2json.py**. (Instructions @ [README](https://github.com/gunrock/io/blob/master/scripts/README.md))

2. Make a folder for output visual representation files. 

3. One can use exsiting scripts to generate different visualization output from JSON files. For example, altair_engines.py generates performance comparison visualization from different graph engines. Below is an example makefile to generate different engines performance comparison figures into .md file into gunrock/doc:

```
    ENGINES_OUTPUTS = output/engines_topc.md \
	output/engines_topc_table_html.md
    PLOTTING_FILES = fileops.py filters.py logic.py
    DEST = "../../gunrock/doc/stats"
    ALL = $(ENGINES_OUTPUTS) \
    all: $(ALL)
    $(ENGINES_OUTPUTS): altair_engines.py $(PLOTTING_FILES)
    		./altair_engines.py
    install: $(ALL)
    		cp $(ALL) $(DEST)
    clean:
    		rm $(ALL)
```

After running these commands, output .md files will be copied into gunrock/doc/stats, in the output directory made in step 2, there will also be .html, .svg, .png, .pdf, .eps and .json output files generated. To start a new python scripts that will output other visualization output, please follow (script @ [altair_engines.py](https://github.com/gunrock/io/scripts/altair_engines.py)).



Reference:

* https://altair-viz.github.io/documentation/displaying.html
* http://wiki.inkscape.org/wiki/index.php/Installing_Inkscape
* https://altair-viz.github.io/installation.html
* http://www.hostingadvice.com/how-to/install-nodejs-ubuntu-14-04/



