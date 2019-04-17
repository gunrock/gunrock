#!/usr/bin/env python

"""
    sfl.py
"""

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
#from matplotlib import pyplot as plt

import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
np.set_printoptions(threshold=np.nan, suppress=True)

# --
# Helpers

def filter_time(nodes, edges, coords, start_time, end_time, time_fmt='%Y-%m-%d %H:%M:%S'):
    """ Sometimes we want to run on only a subset of times """

    nodes.spacetime_group = pd.to_datetime(nodes.spacetime_group, format=time_fmt)
    start_time = datetime.strptime(start_time, time_fmt)
    end_time = datetime.strptime(end_time, time_fmt)

    # Nodes + coords in time period
    sel = np.array((nodes.spacetime_group >= start_time) & (nodes.spacetime_group < end_time))
    nodes = nodes[sel].reset_index(drop=True)
    coords = coords[sel]

    kept_nodes = set(nodes.node_spacetime)
    sel = edges.src.isin(kept_nodes) & edges.trg.isin(kept_nodes)
    edges = edges[sel].reset_index(drop=True)

    return nodes, edges, coords


def make_edges_sequential(nodes, edges):
    """
        SnapVX requires/prefers nodes to have sequential IDs, w/o any gaps
    """
    node_lookup = pd.Series(np.arange(nodes.shape[0]), index=nodes['node_spacetime'])

    edges = np.vstack([
        np.array(node_lookup.loc[edges.src]),
        np.array(node_lookup.loc[edges.trg]),
    ]).T

    # Order edges + remove duplicates
    sel = edges[:,0] >= edges[:,1]
    edges[sel] = edges[sel,::-1]
    edges = np.vstack(set(map(tuple, edges)))

    return edges

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--node-path', type=str, default='./_data/taxi-small/sunday-nodes.tsv')
    parser.add_argument('--edge-path', type=str, default='./_data/taxi-small/sunday-edges.tsv')
    parser.add_argument('--coord-path', type=str, default='./_data/taxi-small/sunday-coords.npy')
    parser.add_argument('--outpath', type=str, default='./_data/taxi-small/sunday-fitted')

    parser.add_argument('--start-time', type=str, default='2011-06-26 12:00:00')
    parser.add_argument('--end-time', type=str, default='2011-06-26 14:00:00')
    #parser.add_argument('--start-time', type=str, default='2011-06-26 08:00:00')
    #parser.add_argument('--end-time', type=str, default='2011-06-26 16:00:00')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # --
    # IO
    print('sfl.py: load data', file=sys.stderr)

    nodes  = pd.read_csv(args.node_path, sep='\t')
    edges  = pd.read_csv(args.edge_path, sep='\t')
    coords = np.load(args.coord_path)

    # --
    # Subset by time
    print('sfl.py: filter time', file=sys.stderr)

    nodes, edges, coords = filter_time(nodes, edges, coords, start_time=args.start_time, end_time=args.end_time)

    # --
    # Convert IDs to sequential ints
    print('sfl.py: sequential ids', file=sys.stderr)

    edges = make_edges_sequential(nodes, edges)
    feats = np.array(nodes.difference)
    print('sfl.py: %d nodes | %d edges' % (nodes.shape[0], edges.shape[0]))

    print(edges.shape)
    print(feats.shape)

    x = edges.tolist()
    x = sorted(x,key=lambda x: (x[0],x[1]))
    edges = np.array(x)
    edges = np.concatenate([np.array([[len(feats),len(edges)]]), edges])

    np.savetxt('e', edges, '%d', delimiter=' ')
    np.savetxt('n', feats, '%.3f', delimiter=' ')
