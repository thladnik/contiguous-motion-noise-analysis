"""Motion receptive fields from CMN responses by reverse correlation.

The method is the two-step non-parametric cluster test of Zhang et al. 2022 as implemented in
``entarchy_vxpy_suite2p.analysis.cmn``: a response-triggered average of the local motion in
direction bins, significance per (patch, bin) against circular shifts of the response, then
significance of spatially connected clusters against the largest cluster of each shift.

All functions take and return plain arrays. Everything that does not depend on the neuron is
computed once per stimulus (:class:`BinnedStimulus`); per neuron the shift null is a handful of
matrix products.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import geometry


# ---------------------------------------------------------------------------------------------
# Response: which samples count, and how much
# ---------------------------------------------------------------------------------------------

def event_weights(dff: np.ndarray, sample_rate: float, selection: np.ndarray,
                  mode: str = 'derivative', kernel_sd: float = 0.5,
                  excluded_percentile: float = 25., noise_mads: float = 1.) -> np.ndarray:
    """Weight of every sample in the response-triggered average; zero outside `selection`.

    The twice-smoothed derivative of dF/F marks the rising phases of the calcium signal, i.e.
    the times the neuron was firing.

    ``mode='binary'`` is the original criterion: derivative > 0, minus the samples in the
    bottom `excluded_percentile` of dF/F, all with weight 1. It selects about a third of the
    samples of a pure-noise trace.

    ``mode='derivative'`` weights each sample by how far its derivative exceeds a noise floor
    of `noise_mads` robust standard deviations (1.4826 MAD of the derivative within the
    selection). Large transients dominate and noise contributes little; the shift null is valid
    for any weights.
    """
    sigma = kernel_sd * sample_rate
    smooth = gaussian_filter1d(np.asarray(dff, dtype=np.float64), sigma, mode='nearest')
    derivative = gaussian_filter1d(np.gradient(smooth), sigma, mode='nearest')
    selection = np.asarray(selection, dtype=bool)

    if mode == 'binary':
        rising = (derivative > 0) & selection
        if not rising.any():
            return np.zeros(len(dff))
        rising &= dff >= np.percentile(dff[rising], excluded_percentile)
        return rising.astype(np.float64)

    if mode == 'derivative':
        inside = derivative[selection]
        noise = 1.4826 * np.median(np.abs(inside - np.median(inside))) if inside.size else 0.
        return np.where(selection, np.clip(derivative - noise_mads * noise, 0., None), 0.)

    raise ValueError(f"mode must be 'binary' or 'derivative', not {mode!r}")


# ---------------------------------------------------------------------------------------------
# Stimulus: direction-binned local motion, once per stimulus
# ---------------------------------------------------------------------------------------------

@dataclass
class BinnedStimulus:
    """Local motion of every CMN sample as speed plus direction bin.

    Equivalent to the one-hot ``(T, P, K)`` representation of the paper (speed in the bin the
    direction falls into, zero elsewhere) at a sixteenth of the memory.
    """
    speed: np.ndarray        # (T, P) float32
    bin_index: np.ndarray    # (T, P) int8
    bin_num: int
    horizontal: np.ndarray   # (P, 3) local frame used for the binning
    vertical: np.ndarray     # (P, 3)

    @classmethod
    def from_vectors(cls, vectors: np.ndarray, centers: np.ndarray, bin_num: int = 16,
                     up: np.ndarray = geometry.UP) -> 'BinnedStimulus':
        """`vectors`: ``(T, P, 3)`` motion vectors of the CMN samples only."""
        horizontal, vertical = geometry.local_frames(centers, up)
        local = geometry.to_local_2d(np.asarray(vectors, dtype=np.float32),
                                     horizontal.astype(np.float32), vertical.astype(np.float32))
        angle = np.arctan2(local[..., 1], local[..., 0])
        bin_index = np.floor((angle + np.pi) / (2 * np.pi) * bin_num).astype(np.int64)
        bin_index = np.clip(bin_index, 0, bin_num - 1).astype(np.int8)  # angle == pi
        speed = np.linalg.norm(local, axis=-1).astype(np.float32)
        return cls(speed, bin_index, bin_num, horizontal, vertical)

    @property
    def sample_num(self) -> int:
        return self.speed.shape[0]

    @property
    def bin_centers(self) -> np.ndarray:
        return -np.pi + (np.arange(self.bin_num) + .5) * 2 * np.pi / self.bin_num

    @property
    def bin_directions(self) -> np.ndarray:
        """Unit vector ``(K, 2)`` of each bin center in the local frame."""
        return np.stack([np.cos(self.bin_centers), np.sin(self.bin_centers)], axis=1)

    def etas(self, weights: np.ndarray) -> np.ndarray:
        """Weighted averages ``(B, P, K)`` for weight rows ``(B, T)`` (or ``(P, K)`` for one
        weight vector): mean speed per direction bin, per patch."""
        single = np.ndim(weights) == 1
        w = np.atleast_2d(weights).astype(np.float32)
        total = w.sum(axis=1)
        w = w / np.where(total > 0, total, 1.)[:, None]
        out = np.empty((w.shape[0], self.speed.shape[1], self.bin_num), dtype=np.float32)
        for k in range(self.bin_num):
            out[:, :, k] = w @ np.where(self.bin_index == k, self.speed, np.float32(0.))
        return out[0] if single else out


def shifted_weights(weights: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Rows ``(B, T)`` of `weights` circularly shifted by each of `shifts` samples."""
    sample_num = len(weights)
    index = (np.arange(sample_num)[None, :] - np.asarray(shifts)[:, None]) % sample_num
    return np.asarray(weights)[index]


# ---------------------------------------------------------------------------------------------
# Step 1: significance per (patch, direction bin)
# ---------------------------------------------------------------------------------------------

def bin_significance(observed: np.ndarray, null: np.ndarray, alpha: float = 0.05):
    """Two-sided significance of `observed` ``(P, K)`` and of every null sample ``(B, P, K)``
    against the null distribution: +1 above the upper ``alpha/2`` tail, -1 below the lower.

    Identical to ranking every sample against all others, from one sort.
    """
    sample_num = null.shape[0]
    ordered = np.sort(null, axis=0)
    upper = ordered[min(int(np.floor(sample_num * (1 - alpha / 2))), sample_num - 1)]
    lower = ordered[max(int(np.ceil(sample_num * alpha / 2)) - 1, 0)]

    def classify(values):
        return (values > upper).astype(np.int8) - (values <= lower).astype(np.int8)

    return classify(observed), classify(null)


# ---------------------------------------------------------------------------------------------
# Step 2: spatially connected clusters
# ---------------------------------------------------------------------------------------------

def find_clusters(significant: np.ndarray, edges: np.ndarray, min_shared: int = 1,
                  values: Optional[np.ndarray] = None):
    """Clusters of adjacent patches that share significant direction bins.

    Two adjacent patches are connected if they have at least `min_shared` significant bins in
    common; the bins shared along any connection are the cluster's members. The result does
    not depend on patch order.

    Returns ``(scores, labels, members)``: the score per cluster (number of member bins, or
    the sum of `values` over them), the cluster number per patch (-1 for none) and the
    ``(P, K)`` bool member mask.
    """
    patch_num = significant.shape[0]
    shared = significant[edges[:, 0]] & significant[edges[:, 1]]
    connected = shared.sum(axis=1) >= min_shared
    left, right, shared = edges[connected, 0], edges[connected, 1], shared[connected]

    members = np.zeros_like(significant, dtype=bool)
    labels = np.full(patch_num, -1, dtype=np.int64)
    if len(left) == 0:
        return np.zeros(0), labels, members

    np.logical_or.at(members, left, shared)
    np.logical_or.at(members, right, shared)

    graph = coo_matrix((np.ones(len(left), dtype=np.int8), (left, right)), shape=(patch_num, patch_num))
    _, component = connected_components(graph, directed=False)
    in_cluster = members.any(axis=1)
    _, labels[in_cluster] = np.unique(component[in_cluster], return_inverse=True)

    per_patch = members.sum(axis=1) if values is None else (values * members).sum(axis=1)
    scores = np.bincount(labels[in_cluster], weights=per_patch[in_cluster])
    return scores, labels, members


# ---------------------------------------------------------------------------------------------
# The whole analysis for one neuron
# ---------------------------------------------------------------------------------------------

@dataclass
class RFResult:
    etas: np.ndarray                  # (P, K) response-triggered mean speed per direction bin
    null_mean: np.ndarray             # (P, K)
    null_std: np.ndarray              # (P, K)
    bin_significance: np.ndarray      # (P, K) int8: +1 / -1 / 0
    cluster_labels: np.ndarray        # (P,) cluster number per patch, -1 for none
    cluster_members: np.ndarray       # (P, K) bool: bins that belong to a cluster
    cluster_scores: np.ndarray        # (C,)
    cluster_p_values: np.ndarray      # (C,)
    cluster_significant: np.ndarray   # (C,) bool
    preferred_vectors: np.ndarray     # (P, 2) in the local frame; zero without significant bins
    preferred_vectors_3d: np.ndarray  # (P, 3) the same as tangential 3D vectors
    rf_patches: np.ndarray            # (P,) bool: patches of significant clusters
    lag: int = 0
    shifts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def z(self) -> np.ndarray:
        return (self.etas - self.null_mean) / np.where(self.null_std > 0, self.null_std, np.inf)

    @property
    def has_receptive_field(self) -> bool:
        return bool(self.cluster_significant.any())


def map_rf(stimulus: BinnedStimulus, weights: np.ndarray, edges: np.ndarray,
           shift_num: int = 1000, min_shift: int = 40, lag: int = 0,
           bin_alpha: float = 0.05, cluster_alpha: float = 0.05, min_shared: int = 1,
           cluster_stat: str = 'mass', seed: Optional[int] = 0) -> RFResult:
    """Receptive field of one neuron.

    `weights` ``(T,)`` are the event weights of the CMN samples, aligned with the rows of
    `stimulus`. `edges` ``(E, 2)`` lists adjacent patches. `lag` [samples] pairs each response
    sample with the stimulus `lag` samples earlier. `min_shift` keeps the circular shifts of
    the null away from the unshifted response (both ends). ``cluster_stat='count'`` scores a
    cluster by its number of member bins (the original statistic), ``'mass'`` by the sum of
    their z-scores. With a fixed `seed` the result is reproducible.
    """
    weights = np.asarray(weights, dtype=np.float64)
    sample_num = stimulus.sample_num
    if len(weights) != sample_num:
        raise ValueError(f'{len(weights)} weights for {sample_num} stimulus samples')
    if weights.sum() <= 0:
        raise ValueError('no response weight inside the CMN samples')
    if sample_num <= 2 * min_shift:
        raise ValueError('too few CMN samples for the requested min_shift')
    if cluster_stat not in ('mass', 'count'):
        raise ValueError(f"cluster_stat must be 'mass' or 'count', not {cluster_stat!r}")

    weights = np.roll(weights, -lag)
    shifts = np.random.default_rng(seed).integers(min_shift, sample_num - min_shift, size=shift_num)

    etas = stimulus.etas(weights)
    null = stimulus.etas(shifted_weights(weights, shifts))
    null_mean, null_std = null.mean(axis=0), null.std(axis=0)

    significance, null_significance = bin_significance(etas, null, bin_alpha)

    def z_of(values):
        return np.clip((values - null_mean) / np.where(null_std > 0, null_std, np.inf), 0., None)

    use_mass = cluster_stat == 'mass'
    scores, labels, members = find_clusters(significance > 0, edges, min_shared,
                                            z_of(etas) if use_mass else None)
    null_max = np.zeros(shift_num)
    for i in range(shift_num):
        null_scores, _, _ = find_clusters(null_significance[i] > 0, edges, min_shared,
                                          z_of(null[i]) if use_mass else None)
        null_max[i] = null_scores.max(initial=0.)

    p_values = (1 + (null_max[None, :] >= scores[:, None]).sum(axis=1)) / (shift_num + 1)
    cluster_significant = p_values < cluster_alpha
    rf_patches = np.isin(labels, np.flatnonzero(cluster_significant))

    # Preferred local motion: excess over the null in the significantly excitatory bins
    excess = np.where(significance > 0, etas - null_mean, 0.)
    preferred = excess @ stimulus.bin_directions
    preferred_3d = geometry.to_global_3d(preferred, stimulus.horizontal, stimulus.vertical)

    return RFResult(etas, null_mean, null_std, significance, labels, members, scores, p_values,
                    cluster_significant, preferred, preferred_3d, rf_patches, lag, shifts)


def lag_profile(stimulus: BinnedStimulus, weights: np.ndarray, lags: Sequence[int],
                shift_num: int = 200, min_shift: int = 40, seed: Optional[int] = 0) -> np.ndarray:
    """Strength of the response-triggered average at each lag [samples]: the RMS z-score over
    all patches and bins (1 for noise). Use it to choose `lag` for :func:`map_rf`."""
    weights = np.asarray(weights, dtype=np.float64)
    shifts = np.random.default_rng(seed).integers(min_shift, stimulus.sample_num - min_shift, size=shift_num)
    null = stimulus.etas(shifted_weights(weights, shifts))
    null_mean, null_std = null.mean(axis=0), null.std(axis=0)
    etas = stimulus.etas(np.stack([np.roll(weights, -int(lag)) for lag in lags]))
    z = (etas - null_mean) / np.where(null_std > 0, null_std, np.inf)
    return np.sqrt((z ** 2).mean(axis=(1, 2)))


# ---------------------------------------------------------------------------------------------
# Egomotion: which self-motion does the receptive field match
# ---------------------------------------------------------------------------------------------

def fit_egomotion(centers: np.ndarray, result: RFResult, axes: Optional[np.ndarray] = None) -> Dict:
    """Best matching translation and rotation flow field for the receptive field.

    The comparison is done on 3D tangential vectors, so it does not depend on the local 2D
    frame (and therefore not on any pitch correction of that frame). Similarities are cosine
    similarities of the whole field over the receptive field patches.
    """
    if axes is None:
        axes = geometry.icosphere_axes(3)
    if not result.rf_patches.any():
        raise ValueError('no receptive field to fit')

    out = {'axes': axes}
    for name, templates in (('translation', geometry.translation_templates(centers, axes)),
                            ('rotation', geometry.rotation_templates(centers, axes))):
        similarities = geometry.field_similarity(templates, result.preferred_vectors_3d, result.rf_patches)
        best = int(np.argmax(similarities))
        out[f'{name}_similarities'] = similarities
        out[f'{name}_best_similarity'] = float(similarities[best])
        out[f'{name}_axis'] = axes[best]
        out[f'{name}_vectors_3d'] = templates[best]
    out['is_translation_selective'] = out['translation_best_similarity'] > out['rotation_best_similarity']
    out['is_rotation_selective'] = out['rotation_best_similarity'] > out['translation_best_similarity']
    return out


def rf_solid_angle(result: RFResult, vertices: Optional[np.ndarray]) -> float:
    """Size of the receptive field [sr] from the actual patch areas."""
    return float(geometry.patch_solid_angles(vertices, len(result.rf_patches))[result.rf_patches].sum())
