# How to map receptive fields with cmn-analysis

This guide goes from a recording of the CMN stimulus and the calcium traces of a set of neurons
to a receptive field per neuron. Everything is plain arrays; the only package-specific object is
the `CMNStimulus`, which holds what was shown.

```
recording  ──►  CMNStimulus  ──►  BinnedStimulus  ──┐
                      │                              ├──►  map_rf  ──►  RFResult  ──►  fit_egomotion
calcium traces  ──►  event_weights  ─────────────────┘
```

## 1. Install

```bash
pip install git+https://github.com/thladnik/contiguous-motion-noise-analysis
```

Needs Python 3.10 or newer, `numpy`, `scipy` and `h5py` (installed automatically).

## 2. Load what was shown

You need, for every CMN presentation, the display frame times and the motion vector of every
sphere patch on every frame, plus the patch centers (and ideally the patch corners for the
mesh adjacency). There are three ways to get there.

### a) From a vxPy 0.1 recording folder

```python
from cmn_analysis import recording_files

stimulus = recording_files.load_cmn_stimulus('path/to/rec')   # reads rec/Display.hdf5
```

Every display phase whose visual name contains `CMN` becomes one segment. Both visuals are
handled: the original one (`<Visual>/motion_vectors_0` at the file root plus `frame_index` per
phase) and the realtime one (`motion_vectors` per phase). Use `name_contains=` if your visuals
are named differently, and `recording_files.phase_table(rec)` to list all phases first.

### b) From a vxpy-next recording

```python
from vxpy.export import SessionData
from cmn_analysis import from_session_data

stimulus = from_session_data(SessionData.load('rec/rec.h5'))
```

### c) From arrays

```python
from cmn_analysis import CMNStimulus

stimulus = CMNStimulus(centers, vertices)       # centers (P, 3) unit vectors, vertices (3 P, 3) or None
stimulus.add_realtime(frame_times, vectors, end_time)                   # vectors (frames, P, 3)
stimulus.add_original(frame_times, frame_index, motion_matrix, end_time)  # matrix (N, P, 3), index (frames,)
```

Call `add_*` once per CMN phase. Times are seconds on the same clock as your imaging times.

## 3. Get the imaging times on the same clock

The analysis pairs each calcium sample with the stimulus frame on screen at that moment, so the
sample times must be on the recording's clock. For a vxPy 0.1 recording where the scanner's
frame clock was logged on a digital input:

```python
times = recording_files.plane_times('path/to/rec', volume_num, plane_num, frames_per_plane)
# (plane_num, volume_num): the acquisition time of every saved image of every plane
```

`frames_per_plane` is the number of raw frames the microscope averaged into each saved image.
For other setups, build the `times` array yourself (one time per sample of your dF/F trace).

## 4. Sample the stimulus at the imaging times

This is done once per stimulus and time base, not per neuron.

```python
from cmn_analysis import rf, geometry

selection, vectors, segment_ids = stimulus.sample(times)
binned = rf.BinnedStimulus.from_vectors(vectors, stimulus.centers)   # bin_num=16 directions
edges = geometry.face_adjacency(stimulus.vertices)                   # which patches touch
```

- `selection` is a boolean mask over `times`: the samples that fall into a CMN phase.
- `vectors` are the motion vectors `(selection.sum(), P, 3)` for exactly those samples.
- `binned` holds the speed and direction bin of every patch on every CMN sample, in a local
  tangent frame. Pass `up=` to `from_vectors` if the fish was pitched relative to the sphere.
- If you have no patch corners, use `geometry.nearest_adjacency(stimulus.centers)` instead.

If a plane has its own time base (volumetric imaging), repeat the sampling per plane and keep
one `binned` per plane. The `edges` are the same for all.

## 5. Map one neuron

```python
weights = rf.event_weights(dff, sample_rate, selection)      # dff: (len(times),), sample_rate [Hz]
result = rf.map_rf(binned, weights[selection], edges)
```

`event_weights` finds the rising phases of the calcium signal and weights each sample by how
strongly the trace was rising there (zero outside the CMN phases). `map_rf` computes the
response-triggered average of the local motion per patch and direction bin, compares it with
1000 circularly shifted copies of the response, and tests spatially connected clusters of
significant bins against the largest cluster of each shift.

The `RFResult` you get back:

| Field | Meaning |
|---|---|
| `has_receptive_field` | Any cluster passed the test. |
| `rf_patches` | `(P,)` bool: the patches of all significant clusters. |
| `preferred_vectors_3d` | `(P, 3)` preferred local motion per patch, as tangential 3D vectors. |
| `bin_significance` | `(P, K)` int8: +1 excitatory, -1 inhibitory, 0 not significant. |
| `cluster_p_values`, `cluster_scores`, `cluster_labels` | Per cluster and per patch. |
| `z` | `(P, K)` z-score of each bin against the shift null. |

Useful parameters of `map_rf`:

- `shift_num=1000`: more shifts, finer p-values, more time. 300 is fine for a first look.
- `lag=0`: pair each response sample with the stimulus `lag` samples earlier. See step 7.
- `min_shift=40`: keep the shifts at least this many samples from the true alignment.
  Use about 4 s worth of samples.
- `seed=0`: fixed seed, reproducible result. Vary it per neuron if you prefer.
- `cluster_stat='mass'`, `event_weights(mode='derivative')`: the defaults. Use
  `'count'` and `'binary'` to reproduce the criteria of the original pipeline.

## 6. Fit the receptive field to egomotion

```python
if result.has_receptive_field:
    fit = rf.fit_egomotion(stimulus.centers, result)
    fit['translation_axis'], fit['translation_best_similarity']
    fit['rotation_axis'], fit['rotation_best_similarity']
    fit['is_translation_selective']
    size_sr = rf.rf_solid_angle(result, stimulus.vertices)
```

The receptive field's preferred vectors are compared with the optic flow of translation along
and rotation about 642 candidate axes, as one cosine similarity over the whole field.

## 7. Choose the response lag

Calcium lags the stimulus. If you do not know the lag, run the analysis for a few lags on the
strongest neurons and keep the lag that yields the most receptive fields:

```python
for lag in (0, 1, 2, 3, 4):   # [samples]
    found = sum(rf.map_rf(binned, w, edges, shift_num=300, lag=lag).has_receptive_field
                for w in strong_weights)
```

`rf.lag_profile(binned, weights, lags)` gives the RMS z-score at each lag for one neuron, which
is quicker but noisier.

## 8. Run many neurons

Everything that depends on the stimulus is in `binned` and `edges`; `map_rf` is independent
per neuron and takes about a second on one core. A `multiprocessing.Pool` with the `binned`
stimulus passed to the workers once (initializer) scales linearly. Set
`OMP_NUM_THREADS=1` in the workers so the BLAS does not oversubscribe the CPU.

A control run with the stimulus reversed in time (`vectors[::-1]` before binning) shows the
false-positive rate of your settings; no neuron can have a true receptive field there.

## 9. The entarchy pipeline

For vxPy 0.1 recordings in an `entarchy_vxpy_suite2p` database, `cmn_analysis.entarchy_adapter`
replaces the two CMN steps of the existing pipeline with the same entry points and attribute
names, so plotting and postprocessing keep working:

```python
from cmn_analysis import entarchy_adapter as adapter

recordings.map_async(adapter.process_recording, sample_rate=10)
rois.map_async(adapter.run_cmn_analysis)          # needs roi['dff_resampled']
```

## Troubleshooting

- **`no phase with a visual named *CMN*`**: the visual class is named differently; pass
  `name_contains=` or check `recording_files.phase_table(rec)`.
- **`no response weight inside the CMN samples`**: the trace is flat during the CMN phases or
  `times` and the stimulus are on different clocks. Check that `selection.sum()` is large.
- **`too few CMN samples for the requested min_shift`**: lower `min_shift` or use a longer
  recording.
- **Everything is significant**: `times` probably has the wrong rate, or dF/F has a slow drift
  that `event_weights` picks up; detrend first.
