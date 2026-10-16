"""Motion receptive fields from contiguous motion noise (CMN) recordings.

Reverse correlation with the two-step cluster test of Zhang et al. 2022, for recordings of
either CMN visual (precomputed motion matrix or realtime generation).

    stimulus_data     what was shown: the motion vectors on the recording's clock
    recording_files   read a vxPy 0.1 recording folder (Display.hdf5, Io.hdf5) directly
    geometry          sphere geometry: tangent frames, patch adjacency, areas, flow templates
    rf                the analysis: event weights, binned stimulus, cluster test, egomotion fit
    entarchy_adapter  drop-in steps for the entarchy_vxpy_suite2p pipeline
"""
__author__ = 'Tim Hladnik'
__copyright__ = 'Copyright 2026'
__license__ = 'MIT'
__version__ = '0.2.0'
__maintainer__ = 'Tim Hladnik'
__email__ = 'tim.github@hladnik.de'
__status__ = 'Development'

from . import geometry, rf, stimulus_data
from .stimulus_data import CMNStimulus, from_session_data

__all__ = ['geometry', 'rf', 'stimulus_data', 'recording_files', 'entarchy_adapter',
           'CMNStimulus', 'from_session_data']
