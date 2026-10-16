# cmn-analysis

Motion receptive fields from recordings of the contiguous motion noise (CMN) stimulus, by
reverse correlation with the two-step cluster test of Zhang & Arrenberg 2019
(J Neurosci Methods 326:108366) and Zhang et al. 2022 (Curr Biol 32).

The package reads what was shown (from a vxPy recording or from plain arrays), weights the
response of each neuron by its calcium events, tests which sphere patches and motion directions
drove it, and fits the significant receptive field against translation and rotation flow fields.
Plain `numpy` and `scipy`, about 1 s per neuron on a CPU for 1000 shifts and 320 patches.

**[How-to guide](docs/how-to.md)**: from a recording folder to receptive fields, step by step.

## Install

```bash
pip install git+https://github.com/thladnik/contiguous-motion-noise-analysis
```

or, for development, from a clone:

```bash
pip install -e .[dev]
```

```bash
pytest
```

## Modules

| Module | What it does |
|---|---|
| `cmn_analysis.stimulus_data` | `CMNStimulus`: the motion vectors on the recording's clock, from either CMN visual. `from_session_data` builds it from a vxpy-next `SessionData`. |
| `cmn_analysis.recording_files` | Reads a vxPy 0.1 recording folder directly: the stimulus from `Display.hdf5`, image times from the frame clock in `Io.hdf5`. |
| `cmn_analysis.geometry` | Sphere geometry: local tangent frames, patch adjacency from the mesh, patch solid angles, egomotion flow templates. |
| `cmn_analysis.rf` | The analysis: `event_weights`, `BinnedStimulus`, `map_rf`, `lag_profile`, `fit_egomotion`. |
| `cmn_analysis.entarchy_adapter` | `process_recording`, `resample_dff` and `run_cmn_analysis` with the entry points and attribute names of the `entarchy_vxpy_suite2p` CMN pipeline. |

## Minimal example

```python
import numpy as np
from cmn_analysis import rf, geometry
from cmn_analysis.stimulus_data import CMNStimulus

# What was shown. centers (P, 3) unit vectors, vertices (3 P, 3) one copy per face.
stimulus = CMNStimulus(centers, vertices)
stimulus.add_realtime(frame_times, motion_vectors, end_time)          # realtime visual, (frames, P, 3)
# stimulus.add_original(frame_times, frame_index, motion_matrix, end_time)   # original visual

# Once per stimulus and imaging time base
selection, vectors, _ = stimulus.sample(times)            # times: the response samples [s]
binned = rf.BinnedStimulus.from_vectors(vectors, stimulus.centers)
edges = geometry.face_adjacency(stimulus.vertices)

# Per neuron
weights = rf.event_weights(dff, sample_rate, selection)
result = rf.map_rf(binned, weights[selection], edges)
if result.has_receptive_field:
    fit = rf.fit_egomotion(stimulus.centers, result)
    print(result.rf_patches.sum(), 'patches;', 'translation' if fit['is_translation_selective'] else 'rotation')
```

## The two CMN visuals

Both are read into the same `CMNStimulus`; nothing downstream knows which one ran.

| Visual | What a recording contains |
|---|---|
| Original (`cmn_redesign.py`) | The whole motion matrix once (`motion_vectors`, `(frames, P, 3)`) and a `frame_index` per display frame. |
| Realtime (`cmn_redesign_realtime.py`) | The motion vectors per display frame (`(P, 3)`). No matrix. |

## Method and what differs from the original pipeline

The method is the two-step cluster test: a response-triggered average of the local motion in
direction bins, significance per (patch, bin) against circular shifts of the response, then
significance of spatially connected clusters against the largest cluster of each shift.

Compared with `entarchy_vxpy_suite2p.analysis.cmn`:

- **Null by matrix product.** The direction-binned stimulus is computed once per stimulus; all
  shifts of one neuron are a few matrix products. No GPU.
- **Step-one thresholds from one sort** instead of ranking every shift against every other.
  Identical output (tested).
- **Clusters from the mesh.** Adjacency comes from shared triangle edges, clusters from a
  connected-components pass. The result does not depend on patch order (tested), no recursion.
- **Egomotion fit in 3D.** Templates and preferred vectors are compared as tangential 3D
  vectors with one cosine similarity over the whole field. A pitch-corrected binning frame
  cannot rotate the data against the templates (tested), patches near a template's
  singularity count little, and no NaN can arise.
- **Event weights.** `event_weights(mode='derivative')` weights samples by the rectified
  calcium derivative above a noise floor. `mode='binary'` is the original criterion.
- **Cluster statistic.** `cluster_stat='mass'` sums z-scores, `'count'` counts bins as before.
- **Preferred vectors** are the excess over the shift null, not raw bin averages.
- **Reproducible.** Shifts come from a seeded generator.
- **Lag.** `map_rf(lag=...)` pairs the response with an earlier stimulus sample;
  `lag_profile` shows which lag carries the signal.
- **Receptive field size** uses the actual patch solid angles.
- **Degenerate frames.** A patch at the pole of the up direction gets a valid local frame.

To reproduce the original criteria: `event_weights(mode='binary')` and
`map_rf(cluster_stat='count')`.

Measured false-positive rate on 60 pure-noise neurons at `cluster_alpha=0.05`: 3 of 60 with
1000 shifts and the defaults.

## Tests

The suite drives the analysis with a field from each visual's own generator (their numerical
cores are copied into `tests/conftest.py`, so no vxPy or GL context is needed) and a simulated
neuron with a planted receptive field. One test exercises `from_session_data` against an
in-memory vxpy-next `SessionData` and is skipped when `vxpy` is not installed.

Not covered: the entarchy adapter against a real database (it runs against stand-in entities),
and `from_session_data` against a recording written by the recorder.

The previous, `caload`-based version of this package is in the git history before the
`new-analysis` branch (`cmn_analysis/functions.py`, `plotting.py`).
