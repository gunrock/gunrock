#!/usr/bin/env python

import vincent # http://vincent.readthedocs.org/en/latest/
               # https://github.com/wrobstory/vincent
import pandas  # http://pandas.pydata.org
import json    # built-in
import os      # built-in

## Load all JSON files into an array of dicts.
## Each array element is one JSON input file (one run).
## Each JSON input file is a dict indexed by attribute.
## If we have more than one JSON object per file:
## http://stackoverflow.com/questions/20400818/python-trying-to-deserialize-multiple-json-objects-in-a-file-with-each-object-s

json_files = [f for f in os.listdir('.')
              if (os.path.isfile(f) and
                  f.endswith(".json") and
                  (f.startswith("BFS") or f.startswith("DOBFS")) and
                  not f.startswith("_"))]
data_unfiltered = [json.load(open(jf)) for jf in json_files]
df = pandas.DataFrame(data_unfiltered)
## All data is now stored in the pandas DataFrame "df".

## Let's add a new column (attribute), conditional on existing columns.
## We'll need this to pivot later.
def setParameters(row):
    return (row['algorithm'] + ', ' +
            ('un' if row['undirected'] else '') + 'directed, ' +
            ('' if row['mark_predecessors'] else 'no ') + 'mark predecessors')
df['parameters'] = df.apply(setParameters, axis=1)

# df.loc[df['mark_predecessors'] & df['undirected'], 'parameters'] = "BFS, undirected, mark predecessors"


## Bar graph, restricted to mark-pred+undirected
## x axis: dataset, y axis: MTEPS
## The following two subsetting operations are equivalent.
df_mteps = df[df['mark_predecessors'] & df['undirected']] # except for BFS
df_mteps = df[df['parameters'] == "BFS, undirected, mark predecessors"]

## draw bar graph
# these next three appear to be equivalent
g_mteps = vincent.Bar(df_mteps,
                      columns=['m_teps'],
                      key_on='dataset') # key_on uses a DataFrame column
                                        # for x-axis values
g_mteps = vincent.Bar(df_mteps.set_index('dataset'),
                      columns=['m_teps'])
g_mteps = vincent.Bar(df_mteps.set_index('dataset')['m_teps'])
## Set plotting parameters for bar graph
g_mteps.axis_titles(x='Dataset', y='MTEPS')
# g_mteps.scales['y'].type = 'log'
g_mteps.colors(brew='Set3')
g_mteps.to_json('_g_mteps.json',
                html_out=True,
                html_path='g_mteps.html')

## Grouped bar graph
## DataFrame needs to be: rows: groups (dataset)
##                        cols: measurements (m_teps, but labeled by categories)
## Each row is a set of measurements grouped together (here, by dataset)
## Each column is an individual measurement (here, mteps)
##
## The pivot changes
##               columns  values
##  [[dataset 1, expt. A, measurement A]]
##  [[dataset 1, expt. B, measurement B]]
## to
##  [[dataset 1, measurement A, measurement B]]

g_grouped = vincent.GroupedBar(df.pivot(index='dataset',
                                        columns='parameters',
                                        values='m_teps'))
g_grouped.axis_titles(x='Dataset', y='MTEPS')
g_grouped.legend(title='Parameters')
g_grouped.colors(brew='Spectral')
# g_grouped.scales['y'].type = 'log'
g_grouped.to_json('_g_grouped.json',
                html_out=True,
                html_path='g_grouped.html')


## OK, let's slurp up the Gunrock paper's BFS data for non-Gunrock engines

## unfortunately gunrock-paper is a private repo, so this doesn't work
datasets = list(df.index)
# dfx = pandas.read_excel("https://github.com/owensgroup/gunrock-paper/raw/master/spread_sheets/comparison.xls", 'BFS')
# when reading, use the dataset names as the index. .T is "transpose"
dfx = pandas.read_excel("comparison.xls", 'BFS', index_col=0).T
dfx.index.name = 'dataset'
# get rid of columns (axis = 1) we don't care about
dfx.drop([
    'Medusa',
    'VertexAPI2',
    'b40c',
    'Davidson SSSP',
    'gpu_BC',
    'Gunrock'], inplace=True, axis=1)
# clean up the crazy Ligra data - X/Y => X
dfx['Ligra'] = dfx['Ligra'].apply(lambda x: float(x.split('/')[0]))
print dfx

## now let's manipulate the Gunrock data.
## Set up the index
df_gunrock = df.set_index('dataset')
## Keep only one parameter set per dataset for comparison
df_gunrock = df_gunrock[df_gunrock['parameters'] == "undirected, no mark predecessors"]
## Keep only "elapsed time" and rename it to Gunrock
df_gunrock = df_gunrock.filter(like='elapsed').rename(columns={'elapsed': 'Gunrock'})
print df_gunrock

# glue 'em together, replace all missing values with 0
dfxg = pandas.concat([dfx, df_gunrock], axis=1).fillna(0)
print dfxg
g_sum = vincent.GroupedBar(dfxg)
g_sum.axis_titles(x='Dataset', y='Elapsed Time')
g_sum.legend(title='Engine')
g_sum.colors(brew='Set2')
g_sum.to_json('_g_sum.json',
              html_out=True,
              html_path='g_sum.html')
