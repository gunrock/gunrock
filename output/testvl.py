#!/usr/bin/env python

import pandas  # http://pandas.pydata.org
import json    # built-in
import os      # built-in
from subprocess import Popen, PIPE, STDOUT # built-in

import matplotlib
import matplotlib.pyplot as plt, mpld3

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
bar = {
    "marktype": "bar",
    "encoding": {
        "y": {"type": "Q","name": "m_teps"},
        "x": {"type": "O","name": "dataset"}
    }
}

# DataFrame cast is only to allow to_dict to run on a df instead of a series
df_mteps_final = pandas.DataFrame(df_mteps.set_index('dataset')['m_teps'])
print df_mteps_final


print df_mteps_final.to_json(orient='split')
print df_mteps_final.to_json(orient='records')
print df_mteps_final.to_json(orient='index')
print df_mteps_final.to_json(orient='columns')
print df_mteps_final.to_json(orient='values')
print df_mteps_final.to_dict(orient='dict')
df_mteps_final.index.name = 'dataset'
df_mteps_final = df_mteps_final.reset_index()
# df_mteps_final.rename(columns={'m_teps': 'y'}, inplace=True)
print df_mteps_final.to_dict(orient='records')
bar["data"] = {"values" : df_mteps_final.to_dict(orient='records')}

# bar now has a full vega-lite description

print("\n\n")
print(bar)
print(json.dumps(bar))

# pipe it through vl2vg to turn it into vega
f_bar = open('_g_bar.json', 'w')
p = Popen(["vl2vg"], stdout=f_bar, stdin=PIPE)
bar_vg = p.communicate(input=json.dumps(bar))[0]
f_bar.close()
# g_bar_vg = json.loads(bar_vg.decode())

# print("\n\n")
# print(g_bar_vg)

## Set plotting parameters for bar graph
# g_bar_vg.scales['y'].type = 'log'
# g_bar_vg.colors(brew='Set3')
# g_bar_vg.to_json('_g_mteps.json',
               # html_out=True,
               # html_path='g_bar_vg.html')

matplotlib.style.use('ggplot')
fig = plt.figure()
fid = df_mteps_final.plot(kind='bar') # not sure how this is embedded into plt.figure()
# mpld3.show()
# mpld3.save_html(fig, "mpld3_bar.html")
