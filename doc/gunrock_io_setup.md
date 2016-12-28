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

With all the dependencies installed, to use gunrock/io, example @ [script](https://github.com/gunrock/io/blob/master/scripts/altair_test.py):

1.  Parses the engine outputs (in txt format) and generates jsons containing important information regarding the output results using **text2json.py**. (Instructions @ [README](https://github.com/gunrock/io/blob/master/scripts/README.md))

2. Make a folder for output visual representation files. 

3. Assuming we are plotting some bfs results from several datasets. In your python scripts, first specify:
	```
	      jsondir = '/json/output/from/run'
	      outputdir = '/where/you/want/to/put/visual/output'
	```
4. Create a list of desired JSON output files, load them as a list of JSON objects and convert them to DataFrame.
	```
	#filter out JSON files needed for this plot
	bfs_json_files = [f for f in os.listdir(jsondir) 
	                      if (os.path.isfile(jsondir + f) and
	                      (os.path.splitext(f)[1] == ".json") and
	                      (os.path.basename(f).startswith("BFS") or
	                       os.path.basename(f).startswith("DOBFS")) and
	                      not os.path.basename(f).startswith("_"))]

	#load list of files as JSON objects
	bfs_data_unfiltered = [json.load(open(jsondir + jf)) for jf in bfs_json_files]

	bfs_df = pandas.DataFrame(bfs_data_unfiltered)

	#One more filtering for this example,
	#some runs in the repo have no dataset. .
	bfs_df = bfs_df[bfs_df['dataset'] != ""]
	```
5. Set chart parameter and load it as a Chart object:
	```
	bfs_chart = Chart(bfs_df).mark_bar().encode(
	    x=X('dataset',
	        axis=Axis(title='Dataset')
	        ),
	    y=Y('m_teps',
	        axis=Axis(title='MTEPS'),
	        scale=Scale(type='log'),
	        ),
	)

	```

	Or to add colors to mark different kind of datasets, set color option,
	```
	def setParameters(row):
	    return (row['algorithm'] + ', ' +
	            ('un' if row['undirected'] else '') + 'directed, ' +
	            ('' if row['mark_predecessors'] else 'no ') + 'mark predecessors')

	bfs_df['parameters'] = bfs_df.apply(setParameters, axis=1)

	bfs_chart = Chart(bfs_df).mark_point().encode(
	    x=X('dataset',
	        axis=Axis(title='Dataset')
	        ),
	    y=Y('m_teps',
	        axis=Axis(title='MTEPS'),
	        scale=Scale(type='log'),
	        ),
	    color='parameters',
	)
```

6. Save chart to different visualization format:
	```
	savefile(bfs_param_t_chart, name='bfs_param_t_chart', fileformat='html',outputdir = outputdir)
	savefile(bfs_param_t_chart, name='bfs_param_t_chart', fileformat='svg', outputdir = outputdir)
	savefile(bfs_param_t_chart, name='bfs_param_t_chart', fileformat='png', outputdir = outputdir)
	savefile(bfs_param_t_chart, name='bfs_param_t_chart', fileformat='pdf', outputdir = outputdir)

	```

To run this example @ [script](https://github.com/gunrock/io/blob/master/scripts/altair_test.py), clone graph_io:

```
cd io/scripts
mkdir example_output
python altair_test.py 
```


Reference:

* https://altair-viz.github.io/documentation/displaying.html
* http://wiki.inkscape.org/wiki/index.php/Installing_Inkscape
* https://altair-viz.github.io/installation.html
* http://www.hostingadvice.com/how-to/install-nodejs-ubuntu-14-04/

