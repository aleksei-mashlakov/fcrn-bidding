import os
import sys

import numpy as np

# Explicitly set path so don't need to run setup.py - if we have multiple copies of the code we would otherwise need
# to setup a separate environment for each to ensure the code pointers are correct.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
)  # noqa

# from pandas.util.testing import assert_frame_equal
# from fcrn_bidding.utils import convert_to_euro


def convert_kW_to_MW(array):
    """Converts kW to MW"""
    return array * 0.001


formula = np.array(0.5)
known_builders = dict()
known_builders["Entity"] = lambda formula: convert_kW_to_MW(formula)
print(known_builders)
