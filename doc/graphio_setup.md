Set up gunrock/graph_io on a server/machine
================

**Grunrock/graph_io Dependencies**

* **Assume the machine has the following env:**
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


**Note**

As of Dec.2016, on midas server:
nodejs, npm, inkspace have been installed in root
local user need to since pandas and numpy:

    pip install --user --upgrade pandas
    pip install --user --upgrade numpy


As of Dec.2016, dependencies version overview:

* python 2.7.6
* gcc 4.8.4
* pandas 0.19.2
* numpy 1.11.3
* altair 1.2.0
* vega 2.6.3
* vega-lite 1.3.1
* inkscape 0.91


Reference:

* https://altair-viz.github.io/documentation/displaying.html
* http://wiki.inkscape.org/wiki/index.php/Installing_Inkscape
* https://altair-viz.github.io/installation.html
* http://www.hostingadvice.com/how-to/install-nodejs-ubuntu-14-04/
