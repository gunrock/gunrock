#!/usr/bin/env python

import vincent
import pandas
import json
import os

json_files = [f for f in os.listdir('.')
              if (os.path.isfile(f) and
                  f.endswith(".json") and
                  not f.startswith("_"))]
data = [json.load(open(jf)) for jf in json_files]
print data
mteps = [d[u'm_teps'] for d in data]
datasets = [d[u'dataset'] for d in data]
print mteps
print datasets

g_mteps = vincent.Bar({'datasets': datasets, 'data': mteps},
                      iter_idx='datasets')
g_mteps.axis_titles(x='Dataset', y='MTEPS')
# g_mteps.scales['y'].type = 'log'
g_mteps.colors(brew='Set3')
g_mteps.to_json('_g_mteps.json',
                html_out=True,
                html_path='g_mteps_template.html')
