#!/usr/bin/env python

"""
    random-values.py
"""

import sys
import numpy as np

vals = np.random.uniform(0, 1, int(sys.argv[1]))
np.savetxt(sys.stdout, vals, fmt='%.8f')