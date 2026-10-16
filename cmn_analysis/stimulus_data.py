"""What was shown: the CMN motion vectors on the recording's clock, from either stimulus.

The two visuals leave different traces in a recording:

- original (``cmn_redesign.py``): the whole precomputed motion matrix is dumped once
  (``motion_vectors``, shape ``(frame_num, P, 3)``) and every display frame records the
  ``frame_index`` into it;
- realtime (``cmn_redesign_realtime.py``): every display frame records the motion vectors
  themselves (``(P, 3)``); there is no matrix.

:class:`CMNStimulus` hides that difference. A stimulus is a list of segments (one per CMN
phase or run), each a series of frame times with a way to look up the vectors; the analysis
only ever calls :meth:`CMNStimulus.sample`.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Segment:
    """One uninterrupted CMN presentation.

    `frame_times` [s] are the display frame times. For a realtime segment `vectors` holds one
    ``(P, 3)`` entry per display frame and `frame_index` is None. For an original segment
    `vectors` is the full motion matrix and `frame_index` maps display frames into it.
    """
    frame_times: np.ndarray
    vectors: np.ndarray
    frame_index: Optional[np.ndarray] = None
    end_time: Optional[float] = None

    def __post_init__(self):
        self.frame_times = np.asarray(self.frame_times, dtype=np.float64).reshape(-1)
        if self.frame_index is not None:
            self.frame_index = np.asarray(self.frame_index).reshape(-1).astype(np.int64)
            if len(self.frame_index) != len(self.frame_times):
                raise ValueError('frame_index and frame_times differ in length')
        elif len(self.vectors) != len(self.frame_times):
            raise ValueError('per-frame vectors and frame_times differ in length')
        if self.end_time is None:
            self.end_time = float(self.frame_times[-1])

    @property
    def kind(self) -> str:
        return 'realtime' if self.frame_index is None else 'original'

    def lookup(self, display_frames: np.ndarray) -> np.ndarray:
        rows = display_frames if self.frame_index is None else self.frame_index[display_frames]
        return self.vectors[rows]


@dataclass
class CMNStimulus:
    centers: np.ndarray                      # (P, 3) unit vectors
    vertices: Optional[np.ndarray] = None    # (3 P, 3): one vertex copy per face, if known
    segments: List[Segment] = field(default_factory=list)

    @property
    def patch_num(self) -> int:
        return len(self.centers)

    def add_original(self, frame_times, frame_index, motion_matrix, end_time=None) -> None:
        self._check(motion_matrix)
        self.segments.append(Segment(frame_times, np.asarray(motion_matrix), frame_index, end_time))

    def add_realtime(self, frame_times, vectors, end_time=None) -> None:
        self._check(vectors)
        self.segments.append(Segment(frame_times, np.asarray(vectors), None, end_time))

    def _check(self, vectors) -> None:
        if np.ndim(vectors) != 3 or np.shape(vectors)[1:] != (self.patch_num, 3):
            raise ValueError(f'expected motion vectors of shape (n, {self.patch_num}, 3), '
                             f'got {np.shape(vectors)}')

    def sample(self, times: np.ndarray, offset: float = 0.) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The vectors on screen at ``times + offset`` [s].

        Returns ``(selection, vectors, segment_ids)``: a bool mask over `times` of the samples
        that fall into a CMN segment, the ``(selection.sum(), P, 3)`` float32 vectors for
        those samples only, and the 1-based segment number per sample (0 outside CMN). The
        frame in effect at a time is the last one displayed at or before it.
        """
        times = np.asarray(times, dtype=np.float64) + offset
        segment_ids = np.zeros(len(times), dtype=np.int64)
        chunks = []
        for number, segment in enumerate(sorted(self.segments, key=lambda s: s.frame_times[0]), start=1):
            inside = (times >= segment.frame_times[0]) & (times <= segment.end_time) & (segment_ids == 0)
            frames = np.searchsorted(segment.frame_times, times[inside], side='right') - 1
            chunks.append((np.flatnonzero(inside), segment.lookup(frames)))
            segment_ids[inside] = number

        selection = segment_ids > 0
        vectors = np.empty((int(selection.sum()), self.patch_num, 3), dtype=np.float32)
        position = np.cumsum(selection) - 1
        for sample_idcs, chunk in chunks:
            vectors[position[sample_idcs]] = chunk
        return selection, vectors, segment_ids


# ---------------------------------------------------------------------------------------------
# vxpy-next recordings (vxpy.export.SessionData)
# ---------------------------------------------------------------------------------------------

def from_session_data(session, visual_class: Optional[str] = None) -> CMNStimulus:
    """Build the stimulus from a vxpy-next recording, ``SessionData.load(path)``.

    Every visual run whose class dumped ``centers`` is taken as a CMN run (restrict with
    `visual_class`, the bare class name). A class that also dumped ``motion_vectors`` is the
    original stimulus and is read through the ``p_frame_index`` track; otherwise the vectors
    come from the ``p_motion_vectors_<geometry>`` track with the matching patch count.
    Times are the session clock in seconds (``t_ns / 1e9``).
    """
    streams = session.streams

    def find(port: str):
        hits = [s for sid, s in streams.items() if sid.rpartition('.')[2] == port]
        return hits[0] if hits else None

    def dumped(class_name: str, key: str):
        stream = find(f'dump_{class_name}__{key}')
        return None if stream is None else np.asarray(stream.data)[-1]

    stimulus = None
    for run in session.runs:
        if run['kind'] != 'visual' or not run.get('class'):
            continue
        class_name = run['class'].replace(':', '.').rpartition('.')[2]
        if visual_class is not None and class_name != visual_class:
            continue
        centers = dumped(class_name, 'centers')
        if centers is None:
            continue

        if stimulus is None:
            stimulus = CMNStimulus(np.asarray(centers, dtype=np.float64), dumped(class_name, 'vertices'))
        elif centers.shape != stimulus.centers.shape or not np.allclose(centers, stimulus.centers, atol=1e-5):
            raise ValueError(f'{class_name} uses a different sphere geometry than the earlier CMN runs; '
                             f'select one with visual_class=')

        start_ns, end_ns = run['start_ns'], run['end_ns']

        def within(stream):
            t_ns = np.asarray(stream.t_ns)
            stop = t_ns[-1] if end_ns is None else end_ns
            return (t_ns >= start_ns) & (t_ns <= stop)

        end_time = None if end_ns is None else end_ns / 1e9
        matrix = dumped(class_name, 'motion_vectors')
        if matrix is not None:
            track = find('p_frame_index')
            if track is None:
                raise ValueError(f'{class_name} dumped a motion matrix but there is no p_frame_index track')
            sel = within(track)
            if not sel.any():
                continue
            stimulus.add_original(track.t_ns[sel] / 1e9, np.asarray(track.data)[sel], matrix, end_time)
        else:
            tracks = [s for sid, s in streams.items()
                      if sid.rpartition('.')[2].startswith('p_motion_vectors')
                      and np.shape(s.data)[1:] == (len(centers), 3)]
            if not tracks:
                raise ValueError(f'{class_name}: neither a dumped motion matrix nor a per-frame '
                                 f'motion vector track for {len(centers)} patches')
            sel = within(tracks[0])
            if not sel.any():
                continue
            stimulus.add_realtime(tracks[0].t_ns[sel] / 1e9, np.asarray(tracks[0].data)[sel], end_time)

    if stimulus is None or not stimulus.segments:
        raise ValueError('no CMN runs found in this recording')
    return stimulus
