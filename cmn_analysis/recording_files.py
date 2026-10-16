"""Read a vxPy 0.1 recording folder directly (``Display.hdf5``, ``Io.hdf5``), without a database."""
import os
from typing import List, Optional

import h5py
import numpy as np

from .stimulus_data import CMNStimulus


def load_cmn_stimulus(rec_path: str, name_contains: str = 'CMN') -> CMNStimulus:
    """The CMN phases of ``<rec_path>/Display.hdf5`` as one stimulus, from either visual.

    The original visual leaves ``<Visual>/motion_vectors_0`` at the file root and ``frame_index``
    in each phase group; the realtime visual leaves ``motion_vectors`` in each phase group.
    Times are vxPy's global clock [s].
    """
    stimulus = None
    matrices = {}
    with h5py.File(os.path.join(rec_path, 'Display.hdf5'), 'r') as f:
        phases = sorted((k for k in f if k.startswith('phase')), key=lambda k: int(k[5:]))
        for key in phases:
            group = f[key]
            visual = str(group.attrs.get('__visual_name', ''))
            if name_contains not in visual:
                continue

            if stimulus is None:
                centers = np.squeeze(f[visual]['centers_0'][:]).astype(np.float64)
                vertices = np.squeeze(f[visual]['vertices_0'][:]) if 'vertices_0' in f[visual] else None
                stimulus = CMNStimulus(centers, vertices)

            frame_times = group['__time'][:, 0]
            end_time = float(group.attrs['__start_time']) + float(group.attrs['__target_duration'])
            if 'motion_vectors' in group:
                stimulus.add_realtime(frame_times, group['motion_vectors'][:], end_time)
            else:
                if visual not in matrices:
                    matrices[visual] = np.squeeze(f[visual]['motion_vectors_0'][:])
                stimulus.add_original(frame_times, group['frame_index'][:, 0], matrices[visual], end_time)

    if stimulus is None:
        raise ValueError(f'no phase with a visual named *{name_contains}* in {rec_path}')
    return stimulus


def phase_table(rec_path: str) -> List[dict]:
    """``index``, ``visual``, ``start`` and ``end`` [s] of every display phase, plus its scalar
    parameters (first value of each one-column dataset)."""
    rows = []
    with h5py.File(os.path.join(rec_path, 'Display.hdf5'), 'r') as f:
        for key in sorted((k for k in f if k.startswith('phase')), key=lambda k: int(k[5:])):
            group = f[key]
            start = float(group.attrs['__start_time'])
            row = {'index': int(key[5:]), 'visual': str(group.attrs.get('__visual_name', '')),
                   'start': start, 'end': start + float(group.attrs['__target_duration'])}
            for name, dataset in group.items():
                if not name.startswith('__') and dataset.ndim == 2 and dataset.shape[1] <= 3 and len(dataset):
                    row[name] = np.squeeze(dataset[0])
            rows.append(row)
    return rows


def frame_clock_edges(rec_path: str, signal: str = 'di_frame_sync') -> np.ndarray:
    """Times [s] of every edge (rising and falling) of the scanner's frame clock in ``Io.hdf5``.
    The line toggles once per acquired raw frame."""
    with h5py.File(os.path.join(rec_path, 'Io.hdf5'), 'r') as f:
        level = f[signal][:, 0].astype(np.int8)
        times = f[f'{signal}_time'][:, 0]
    return times[np.flatnonzero(np.diff(level) != 0) + 1]


def plane_times(rec_path: str, volume_num: int, plane_num: int, frames_per_plane: int = 1,
                frame_rate: Optional[float] = None) -> np.ndarray:
    """Acquisition time ``(plane_num, volume_num)`` of every saved image on vxPy's clock.

    A saved image of plane ``l`` in volume ``v`` averages the raw frames
    ``(v * plane_num + l) * frames_per_plane + 0 .. frames_per_plane - 1``; its time is the middle
    of that span. Raw frame ``k`` starts at the k-th clock edge.
    """
    edges = frame_clock_edges(rec_path)
    raw_needed = volume_num * plane_num * frames_per_plane
    if len(edges) < raw_needed:
        raise ValueError(f'{len(edges)} clock edges for {raw_needed} raw frames')
    period = np.median(np.diff(edges)) if frame_rate is None else 1 / frame_rate
    first = np.arange(volume_num * plane_num) * frames_per_plane
    middle = edges[first] + (edges[first + frames_per_plane - 1] - edges[first]) / 2 + period / 2
    return middle.reshape((volume_num, plane_num)).T
