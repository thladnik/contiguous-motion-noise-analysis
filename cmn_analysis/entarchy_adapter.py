"""Drop-in steps for the ``entarchy_vxpy_suite2p`` pipeline (vxPy 0.1 recordings).

Same entry points as ``entarchy_vxpy_suite2p.analysis.cmn.functions`` and the same attribute
names wherever the meaning is unchanged, so the existing plotting and postprocessing keep
working::

    recordings.map_async(adapter.process_recording, sample_rate=10)
    rois.map_async(adapter.run_cmn_analysis)          # needs roi['dff_resampled']

Both stimuli are handled. In a vxPy 0.1 recording the original visual leaves
``display/<Visual>/motion_vectors_0`` on the recording plus ``frame_index`` on each phase; the
realtime visual leaves ``motion_vectors`` on each phase instead.

Only the entity access that the existing pipeline already uses is relied on: ``entity[key]``,
``rec.phases``, ``rec.layers``, ``rec.rois``, ``rec.animal``, ``rec.sole_imaging()``,
``roi.recording`` and ``roi.layer``.
"""
from collections import OrderedDict

import numpy as np

from . import geometry, rf
from .stimulus_data import CMNStimulus


def _get(entity, key, default=None):
    try:
        value = entity[key]
    except (KeyError, AttributeError):
        return default
    return default if value is None else value


def up_direction(rec) -> np.ndarray:
    """Fish-up in stimulus coordinates: z, corrected by the pitch set during registration."""
    pitch = _get(rec.animal, 'ants/init_x_rotation')
    if pitch is None:
        return geometry.UP.copy()
    a = np.deg2rad(pitch)
    rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return geometry.UP @ rotation


def stimulus_from_recording(rec) -> CMNStimulus:
    """All CMN phases of a recording as one stimulus, whichever visual produced them."""
    stimulus = None
    for phase in rec.phases:
        visual_name = _get(phase, 'display/__visual_name', '')
        if 'CMN' not in visual_name:
            continue

        centers = np.squeeze(rec[f'display/{visual_name}/centers_0'][:])
        if stimulus is None:
            vertices = _get(rec, f'display/{visual_name}/vertices_0')
            stimulus = CMNStimulus(centers.astype(np.float64),
                                   None if vertices is None else np.squeeze(vertices[:]))
        elif centers.shape != stimulus.centers.shape:
            raise ValueError(f'{rec}: CMN phases with different sphere geometries')

        frame_times = np.squeeze(phase['__time'])
        start = phase['display/__start_time']
        end_time = start + phase['display/__target_duration']

        per_frame = _get(phase, 'motion_vectors')
        if per_frame is not None:
            stimulus.add_realtime(frame_times, np.asarray(per_frame), end_time)
        else:
            matrix = np.squeeze(rec[f'display/{visual_name}/motion_vectors_0'][:])
            stimulus.add_original(frame_times, np.squeeze(phase['frame_index']), matrix, end_time)
    return stimulus


def process_recording(rec, sample_rate: float = 10., radial_bin_num: int = 16):
    """Resampled time base and, per layer, the CMN samples with their motion vectors."""
    if len(rec.imaging) == 0:
        print(f'{rec} has no imaging data; skipping')
        return

    ca_times = rec.sole_imaging()['frame_times']
    time_resampled = np.arange(ca_times[0], ca_times[-1], 1 / sample_rate)
    rec['time_resampled'] = time_resampled
    rec['sample_rate'] = sample_rate

    stimulus = stimulus_from_recording(rec)
    rec.rois['has_cmn_dataset'] = stimulus is not None
    if stimulus is None:
        return

    for layer in rec.layers:
        selection, vectors, phase_ids = stimulus.sample(time_resampled, offset=layer['t_offset'])
        layer['cmn_phase_selection'] = selection
        layer['cmn_phase_ids'] = phase_ids
        layer['cmn_motion_vectors_3d'] = vectors  # CMN samples only, float32

    rec['positions'] = stimulus.centers
    if stimulus.vertices is not None:
        rec['patch_corners'] = stimulus.vertices
        rec['patch_adjacency'] = geometry.face_adjacency(stimulus.vertices)
    else:
        rec['patch_adjacency'] = geometry.nearest_adjacency(stimulus.centers)
    rec['patch_solid_angles'] = geometry.patch_solid_angles(stimulus.vertices, stimulus.patch_num)
    rec['cmn_up_direction'] = up_direction(rec)
    rec['cmn_stimulus_kinds'] = sorted({segment.kind for segment in stimulus.segments})

    edges = np.linspace(-np.pi, np.pi, radial_bin_num + 1)
    rec['radial_bin_num'] = radial_bin_num
    rec['radial_bin_edges'] = edges
    rec['radial_bin_centers'] = edges[:-1] + np.diff(edges) / 2
    rec['egomotion_axes_opts'] = geometry.icosphere_axes(3)


# The binned stimulus is identical for all ROIs of a layer; keep the last few per worker process
_binned_cache: 'OrderedDict' = OrderedDict()


def _binned_stimulus(roi) -> rf.BinnedStimulus:
    layer, rec = roi.layer, roi.recording
    key = (getattr(rec, 'id', None), getattr(layer, 'id', None))
    if None not in key and key in _binned_cache:
        return _binned_cache[key]
    binned = rf.BinnedStimulus.from_vectors(layer['cmn_motion_vectors_3d'], rec['positions'],
                                            rec['radial_bin_num'], rec['cmn_up_direction'])
    if None not in key:
        _binned_cache[key] = binned
        while len(_binned_cache) > 2:
            _binned_cache.popitem(last=False)
    return binned


def run_cmn_analysis(roi, event_mode: str = 'derivative', lag_seconds: float = 0.,
                     bootstrap_num: int = 1000, bernoulli_alpha: float = 0.05,
                     cluster_alpha: float = 0.05, sign_radial_bin_threshold: int = 1,
                     cluster_stat: str = 'mass', seed: int = 0, save_debug_data: bool = False):
    """Receptive field and egomotion fit of one ROI. ``event_mode='binary'`` and
    ``cluster_stat='count'`` reproduce the criteria of the original pipeline."""
    rec, layer = roi.recording, roi.layer
    sample_rate = rec['sample_rate']
    selection = layer['cmn_phase_selection']
    dff = roi['dff_resampled']

    weights = rf.event_weights(dff, sample_rate, selection, mode=event_mode)
    roi['signal_weights'] = weights
    roi['signal_selection'] = weights > 0
    roi['signal_length'] = int((weights > 0).sum())
    roi['signal_proportion'] = float((weights > 0).sum() / max(selection.sum(), 1))
    if weights.sum() <= 0:
        roi['has_receptive_field'] = False
        return

    binned = _binned_stimulus(roi)
    result = rf.map_rf(binned, weights[selection], rec['patch_adjacency'],
                       shift_num=bootstrap_num, min_shift=int(4 * sample_rate),
                       lag=int(round(lag_seconds * sample_rate)), bin_alpha=bernoulli_alpha,
                       cluster_alpha=cluster_alpha, min_shared=sign_radial_bin_threshold,
                       cluster_stat=cluster_stat, seed=seed)

    clusters = [tuple(np.flatnonzero(result.cluster_labels == c)) for c in range(len(result.cluster_scores))]
    roi['radial_bin_significances'] = result.bin_significance
    roi['preferred_vectors'] = result.preferred_vectors
    roi['cluster_unique_patch_indices'] = clusters
    roi['cluster_scores'] = result.cluster_scores
    roi['cluster_p_values'] = result.cluster_p_values
    roi['cluster_significant_indices'] = np.flatnonzero(result.cluster_significant)
    roi['significant_patch_indices'] = np.flatnonzero(result.rf_patches)
    roi['rf_size_sr'] = float(rec['patch_solid_angles'][result.rf_patches].sum())
    roi['has_receptive_field'] = result.has_receptive_field
    if save_debug_data:
        roi['radial_bin_etas'] = result.etas
        roi['radial_bin_null_mean'] = result.null_mean
        roi['radial_bin_null_std'] = result.null_std

    if not result.has_receptive_field:
        return

    fit = rf.fit_egomotion(rec['positions'], result, rec['egomotion_axes_opts'])
    for name in ('translation', 'rotation'):
        roi[f'{name}_similarities'] = fit[f'{name}_similarities']
        roi[f'{name}_best_similarity'] = fit[f'{name}_best_similarity']
        roi[f'{name}_axis'] = fit[f'{name}_axis']
        # 2D in the same local frame as preferred_vectors, as the plotting expects
        roi[f'{name}_vectors'] = geometry.to_local_2d(fit[f'{name}_vectors_3d'],
                                                      binned.horizontal, binned.vertical)
    roi['is_translation_selective'] = bool(fit['is_translation_selective'])
    roi['is_rotation_selective'] = bool(fit['is_rotation_selective'])
