import pandas as pd
from typing import List, Tuple, Union

import numpy as np
import scipy
import torch

import helper

from caload.schemata.ca_imaging_s2p_vxpy import *


def calculate_dff(roi: Roi, window_size: int = 120, percentile: int = 10):

    # Calculate DFF
    imaging_rate = roi.recording['imaging_rate']
    window_size = int(window_size * imaging_rate)
    if window_size % 2 == 0:
        window_size += 1
    half_window_size = int((window_size - 1) // 2)

    fluorescence = roi['fluorescence']
    # Pad and fill
    f_padded = np.pad(fluorescence, half_window_size, mode='empty')
    f_padded[:half_window_size] = np.median(fluorescence[:half_window_size])
    f_padded[-half_window_size:] = np.median(fluorescence[-half_window_size:])
    # Calculate for each signal datapoint
    dff = np.zeros(fluorescence.shape)
    for i in range(dff.shape[0]):
        fsub = f_padded[i:i + window_size]
        fmean = np.mean(fsub[fsub < np.percentile(fsub, percentile)])
        dff[i] = (fluorescence[i] - fmean) / fmean
    # Save
    roi['dff'] = dff


def process_recording(rec: Recording, sample_rate: float = 10., radial_bin_num: int = 16):
    # Get data
    ca_times = rec['ca_times']
    time_resampled = np.arange(ca_times[0], ca_times[-1], 1 / sample_rate)
    repeat_num = rec['metadata/repeat_num']

    # Get phases dataframe
    # !! IMPORTTANT NOTE: this approach only works if the stimulus order between repeats is identical
    phase_df = pd.DataFrame([p.attributes for p in rec.phases])
    phase_df = phase_df.sort_values(['phase_id_int'])

    # Translation
    translation_df = phase_df[phase_df['__visual_name'] == 'TranslationGrating'].copy()
    # Create continuous ID for calculating repeat_index
    translation_repeat_len = len(translation_df) // repeat_num
    translation_df['continuous_id'] = list(range(len(translation_df)))
    translation_df['repeat_num'] = translation_df['continuous_id'] // translation_repeat_len
    translation_df['repeat_index'] = translation_df['continuous_id'] % translation_repeat_len

    # Save
    rec['translation_dataframe'] = translation_df
    rec['translation_repeat_len'] = translation_repeat_len

    # Rotation
    rotation_df = phase_df[phase_df['__visual_name'] == 'RotationGrating'].copy()
    # Create continuous ID for calculating repeat_index
    rotation_repeat_len = len(rotation_df) // repeat_num
    rotation_df['continuous_id'] = list(range(len(rotation_df)))
    rotation_df['repeat_num'] = rotation_df['continuous_id'] // rotation_repeat_len
    rotation_df['repeat_index'] = rotation_df['continuous_id'] % rotation_repeat_len

    # Save
    rec['rotation_dataframe'] = rotation_df
    rec['rotation_repeat_len'] = rotation_repeat_len

    # Go through all CMN phases and add CMN data to resampled time domain
    cmn_phase_ids_original = np.zeros_like(ca_times, dtype=int)
    cmn_phase_selection_original = np.zeros_like(ca_times, dtype=bool)
    cmn_phase_ids = np.zeros_like(time_resampled, dtype=int)
    cmn_phase_selection = np.zeros_like(time_resampled, dtype=bool)
    cmn_motion_vectors_3d = None
    cmn_motion_vectors_3d_original = None
    positions = None
    cmn_phase_counter = 0
    for phase in rec.phases:
        if 'CMN' not in phase['__visual_name']:
            continue

        # Increment counter
        cmn_phase_counter += 1

        cmn_start_time = phase['__start_time']
        cmn_end_time = cmn_start_time + phase['__target_duration']

        # Add CMN phase to selection
        # Original Ca sample rate
        selection_original = (cmn_start_time <= ca_times) & (ca_times <= cmn_end_time)
        cmn_phase_selection_original |= selection_original
        cmn_phase_ids_original[selection_original] = cmn_phase_counter
        # Resampled
        selection = (cmn_start_time <= time_resampled) & (time_resampled <= cmn_end_time)
        cmn_phase_selection |= selection
        cmn_phase_ids[selection] = cmn_phase_counter

        # Get CMN base data
        cmn_name = phase['__visual_name']
        positions = rec[f'display/{cmn_name}/centers_0'][:].squeeze()
        patch_corners = rec[f'display/{cmn_name}/vertices_0'][:].squeeze()
        patch_indices = rec[f'display/{cmn_name}/indices_0'][:].squeeze()

        # Get CMN phase data
        frame_indices = phase['frame_index'].squeeze()
        frame_times = phase['__time'].squeeze()
        motion_vectors_full = rec[f'display/{cmn_name}/motion_vectors_0'][:].squeeze()

        # Find corresponding CMN indices
        indices = [np.argmin(np.abs(frame_times - t)) for t in time_resampled[selection]]

        # Update motion vectors
        if cmn_motion_vectors_3d is None:
            cmn_motion_vectors_3d = np.nan * np.ones(time_resampled.shape + (motion_vectors_full.shape[1:]))

        cmn_motion_vectors_3d[selection] = motion_vectors_full[frame_indices[indices]]

        # ORIGINAL timeline

        # Find corresponding CMN indices
        indices = [np.argmin(np.abs(frame_times - t)) for t in ca_times[selection_original]]

        # Update motion vectors
        if cmn_motion_vectors_3d_original is None:
            cmn_motion_vectors_3d_original = np.nan * np.ones(ca_times.shape + (motion_vectors_full.shape[1:]))

        cmn_motion_vectors_3d_original[selection_original] = motion_vectors_full[frame_indices[indices]]

    if positions is None:
        raise Exception(f'No CMN data found in {rec}')

    # Define up direction
    vertical_up_direction = np.array([0, 0, 1])
    if 'ants/init_x_rotation' in rec.animal:
        # Use the manually adjusted pitch during registration alignment to correct fish rotation
        pitch_rot = np.deg2rad(rec.animal['ants/init_x_rotation'])
        R = np.array([[np.cos(pitch_rot), 0, np.sin(pitch_rot)],
                      [0, 1, 0],
                      [-np.sin(pitch_rot), 0, np.cos(pitch_rot)]])
        vertical_up_direction = vertical_up_direction @ R

    rec['time_resampled'] = time_resampled
    rec['sample_rate'] = sample_rate
    rec['positions'] = positions
    rec['patch_corners'] = patch_corners
    rec['patch_indices'] = patch_indices
    rec['cmn_phase_selection_original'] = cmn_phase_selection_original
    rec['cmn_phase_ids_original'] = cmn_phase_ids_original
    rec['cmn_phase_selection'] = cmn_phase_selection
    rec['cmn_phase_ids'] = cmn_phase_ids
    rec['cmn_motion_vectors_3d'] = cmn_motion_vectors_3d
    rec['cmn_motion_vectors_2d'] = project_to_local_2d_vectors(positions, cmn_motion_vectors_3d, vertical_up_direction)
    rec['cmn_motion_vectors_3d_original'] = cmn_motion_vectors_3d_original
    rec['cmn_motion_vectors_2d_original'] = project_to_local_2d_vectors(positions, cmn_motion_vectors_3d_original, vertical_up_direction)

    # Set radial bins
    radial_bin_edges = np.linspace(-np.pi, np.pi, radial_bin_num + 1)
    radial_bin_centers = radial_bin_edges[:-1] + (radial_bin_edges[1] - radial_bin_edges[0]) / 2

    rec['radial_bin_num'] = radial_bin_centers.shape[0]
    rec['radial_bin_edges'] = radial_bin_edges
    rec['radial_bin_centers'] = radial_bin_centers

    # Check radial histograms motion matrix
    # cmn_motion_vectors_2d = project_to_local_2d_vectors(positions, cmn_motion_vectors_3d[cmn_phase_selection])
    # norms, etas = calculate_local_directions(cmn_motion_vectors_2d, radial_bin_edges)
    # fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    # plot.plot_radial_histograms(ax, positions, etas, radial_bin_edges)
    # plt.show()

    # Watch motion matrix quiver animation
    # plot.animate_motion_matrix(time_resampled[cmn_phase_selection], positions,
    #                       cmn_motion_vectors_3d[cmn_phase_selection], sample_rate)

    # Calculate pairwise distances between all patches and select the three adjacent ones for each patch
    patch_num = positions.shape[0]
    pairwise_distances = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
    closest_3_position_indices = np.zeros((patch_num, 3), dtype=np.int64)
    for patch_idx in range(patch_num):
        closest_3_position_indices[patch_idx] = np.argsort(pairwise_distances[patch_idx])[1:4]
    rec['clostest_3_position_indices'] = closest_3_position_indices

    # Calculate egomotion axes and corresponding local motion fields
    icosphere = helper.IcosahedronSphere(subdiv_lvl=3)
    egomotion_axes_opts = icosphere.get_vertices()
    rec['egomotion_axes_opts'] = egomotion_axes_opts

    # Local motion for translation templates
    egomotion_translation_local_motion = np.zeros((egomotion_axes_opts.shape[0], positions.shape[0], 2))
    for i, translation_dir in enumerate(egomotion_axes_opts):
        vnorms = np.array([0, 0, 1]) - positions * np.dot(positions, np.array([0, 0, 1]))[:, None]
        vnorms /= np.linalg.norm(vnorms, axis=1)[:, None]

        hnorms = -crossproduct(vnorms, positions)
        hnorms /= np.linalg.norm(hnorms, axis=1)[:, None]

        # Calculate 2d motion vectors in coordinate system defined by local horizontal and vertical norms
        egomotion_translation_local_motion[i] = np.array([np.sum(translation_dir * hnorms, axis=1),
                                                          np.sum(translation_dir * vnorms, axis=1)]).T

    rec['egomotion_translation_local_motion'] = egomotion_translation_local_motion

    # Local motion for rotation templates
    egomotion_rotation_local_motion = np.zeros((egomotion_axes_opts.shape[0], positions.shape[0], 2))
    for i, axis_vec in enumerate(egomotion_axes_opts):

        vnorms = np.array([0, 0, 1]) - positions * np.dot(positions, np.array([0, 0, 1]))[:, None]
        vnorms /= np.linalg.norm(vnorms, axis=1)[:, None]

        hnorms = -crossproduct(vnorms, positions)
        hnorms /= np.linalg.norm(hnorms, axis=1)[:, None]

        rotvecs = -crossproduct(axis_vec, positions)
        rotvecs /= np.linalg.norm(rotvecs, axis=1)[:, None]

        # Calculate 2d motion vectors in coordinate system defined by local horizontal and vertical norms
        egomotion_rotation_local_motion[i] = np.array([np.sum(rotvecs * hnorms, axis=1), np.sum(rotvecs * vnorms, axis=1)]).T
        egomotion_rotation_local_motion[i] *= (1 - np.abs(np.array([np.dot(v, axis_vec) for v in positions])))[:, None]

    rec['egomotion_rotation_local_motion'] = egomotion_rotation_local_motion


def calculate_autocorrelations(roi: Roi):
    repeat_num = roi.recording['metadata/repeat_num']
    dff = roi['dff']
    ca_times = roi.recording['ca_times']
    time_resampled = roi.recording['time_resampled']

    # Interpolate DFF to new time
    roi['dff_resampled'] = scipy.interpolate.interp1d(ca_times, dff, kind='nearest')(time_resampled)

    for motion_type in ['translation', 'rotation']:

        phase_df = roi.recording[f'{motion_type}_dataframe']
        repeat_len = roi.recording[f'{motion_type}_repeat_len']

        # Calculate phase max DFFs
        dff_phase_max = np.zeros((repeat_num, repeat_len))
        for _, phase in phase_df.iterrows():
            dff_p = dff[phase['ca_start_index']:phase['ca_end_index']]
            phase['dff'] = dff_p
            dff_phase_max[phase['repeat_num'], phase['repeat_index']] = np.mean(dff_p)
        roi[f'{motion_type}_dff_phase_mean'] = dff_phase_max

        # Calculate autocorrelation
        roi[f'{motion_type}_autocorrelation'] = np.mean(np.corrcoef(dff_phase_max)[~np.eye(repeat_num, dtype=bool)])


def detect_events_with_derivative(roi: Roi, excluded_percentile: int = 25, kernel_sd: float = 0.5):
    cmn_selection = roi.recording['cmn_phase_selection']
    dff = roi['dff_resampled']
    sample_rate = roi.recording['sample_rate']

    kernel_dts = 10 * sample_rate
    kernel_t = np.linspace(-5, 5, kernel_dts)
    norm_kernel = scipy.stats.norm.pdf(kernel_t, scale=kernel_sd)

    # Smoothen DFF
    dff_plot_pad = np.zeros(dff.shape[0] + kernel_dts - 1)
    dff_plot_pad[kernel_dts // 2:-kernel_dts // 2 + 1] = dff
    dff_conv = scipy.signal.convolve(dff_plot_pad, norm_kernel, mode='valid', method='fft')

    # Calculate derivative
    diff = np.diff(dff_conv, append=[0])

    # Smoothen derivative
    diff1_pad = np.zeros(diff.shape[0] + kernel_dts - 1)
    diff1_pad[kernel_dts // 2:-kernel_dts // 2 + 1] = diff
    diff1_conv = scipy.signal.convolve(diff1_pad, norm_kernel, mode='valid', method='fft')

    # Calculate signal based on smoothened derivative
    signal_selection = (diff1_conv > 0) & cmn_selection
    # Exclude bottom percentile
    signal_selection &= dff >= np.percentile(dff[signal_selection], excluded_percentile)

    # Final selection
    roi['signal_selection'] = signal_selection

    signal_length = sum(signal_selection)
    roi['signal_length'] = signal_length
    roi['signal_proportion'] = signal_length / sum(cmn_selection)
    roi['signal_dff_mean'] = np.mean(dff[signal_selection])


def calculate_reverse_correlations(roi: Roi,
                                   bootstrap_num: int = 1000,
                                   use_gpu: bool = False,
                                   use_cmn_phase_id: int = None):
    """
    roi: Roi entity object to operate on
    bootstrap_num: int number of bootstrap iterations (default: 1000)
    use_gpu: bool whether to use GPU or not. True for GPU and False for CPU only (default: False)
    """

    # ROI data
    signal_selection = roi['signal_selection']

    # Only use specified cmn segment
    postfix = ''
    if use_cmn_phase_id is not None:
        postfix = f'_{use_cmn_phase_id}'
        signal_selection &= roi.recording['cmn_phase_ids'] == use_cmn_phase_id
        roi[f'signal_selection{postfix}'] = signal_selection

    # Recording data
    sample_rate = roi.recording['sample_rate']
    radial_bin_edges = roi.recording['radial_bin_edges']
    motion_vectors_2d = roi.recording['cmn_motion_vectors_2d']
    cmn_phase_selection = roi.recording['cmn_phase_selection']

    radial_bin_norms, radial_bin_etas = calculate_local_directions(motion_vectors_2d[signal_selection],
                                                                   radial_bin_edges)

    roi[f'radial_bin_etas{postfix}'] = radial_bin_etas

    # print('Bootstrap local directions (step 1)')

    # Select motion matrix based on circularly permutated signal
    min_frame_shift = 4 * sample_rate
    max_frame_shift = int(cmn_phase_selection.sum() - min_frame_shift)
    frame_shifts = np.random.randint(min_frame_shift, max_frame_shift, size=(bootstrap_num,))
    # print(f'Frame shift range set to {max_frame_shift - min_frame_shift} frames @ {sample_rate} FPS')
    bs_radial_bin_etas = np.zeros((bootstrap_num,) + radial_bin_etas.shape)
    signal_within_cmn_selection = signal_selection[cmn_phase_selection]

    # Compute
    if use_gpu:

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Convert numpy arrays to PyTorch tensors
        motion_vectors_2d_tensor = torch.tensor(motion_vectors_2d, dtype=torch.float32).to(device)
        radial_bin_edges_tensor = torch.tensor(radial_bin_edges, dtype=torch.float32).to(device)

        # Prepare for bootstrapping
        cmn_phase_selection_tensor = torch.Tensor(cmn_phase_selection).bool().to(device)
        signal_within_cmn_selection_tensor = torch.Tensor(signal_within_cmn_selection).bool().to(device)
        # Create output tensor
        bs_radial_bin_etas_tensor = torch.zeros((bootstrap_num, *radial_bin_etas.shape), dtype=torch.float32).to(device)
        for i in range(bootstrap_num):
            # Circularly permutate signal
            perm_signal_selection = torch.roll(signal_within_cmn_selection_tensor, frame_shifts[i])

            # Get motion vectors of permutated signal
            bs_motion_vectors_tensor = motion_vectors_2d_tensor[cmn_phase_selection_tensor][perm_signal_selection]

            # Calculate vector ETAs for each local radial bin
            _, bs_bin_etas = calculate_local_directions_torch(bs_motion_vectors_tensor, radial_bin_edges_tensor)

            # Save ETAs
            bs_radial_bin_etas_tensor[i] = bs_bin_etas

        # Copy resulting output tensor back to CPU and convert to numpy
        bs_radial_bin_etas = bs_radial_bin_etas_tensor.cpu().numpy()

    else:

        for i in range(bootstrap_num):
            # Circularly permutate signal
            perm_signal_selection = np.roll(signal_within_cmn_selection, frame_shifts[i])

            # Get motion vectors of permutated signal
            bs_motion_vectors = motion_vectors_2d[cmn_phase_selection][perm_signal_selection]

            # Calculate vector ETAs for each local radial bin
            bs_radial_bin_etas[i] = calculate_local_directions(bs_motion_vectors, radial_bin_edges)[1]

    # Save bootstrapped ETAs
    roi[f'bs_radial_bin_etas{postfix}'] = bs_radial_bin_etas


def run_2step_ncb_test(roi: Roi,
                       bernoulli_alpha: float = 0.05,
                       cluster_alpha: float = 0.05,
                       sign_radial_bin_threshold: int = 1,
                       postfix: str = ''):
    # Roi
    radial_bin_etas = roi[f'radial_bin_etas{postfix}']
    bs_radial_bin_etas = roi[f'bs_radial_bin_etas{postfix}']
    bootstrap_num = bs_radial_bin_etas.shape[0]

    # Calculate ETA significances (1st bootstrap)

    # For original ETAs
    cdf_values = ((radial_bin_etas > bs_radial_bin_etas).sum(axis=0) / bootstrap_num)
    greater_than = cdf_values > 1 - bernoulli_alpha / 2
    less_than = cdf_values < bernoulli_alpha / 2
    radial_bin_p_values = cdf_values.copy()
    radial_bin_p_values[greater_than] = 1 - cdf_values[greater_than]

    radial_bin_significances = np.zeros_like(radial_bin_p_values, dtype=np.int8)
    radial_bin_significances[greater_than] = 1
    radial_bin_significances[less_than] = -1

    # roi[f'radial_bin_p_values{postfix}'] = radial_bin_p_values
    roi[f'radial_bin_significances{postfix}'] = radial_bin_significances

    #  For bootstrapped ETAs
    bs_radial_bin_significances = np.zeros_like(bs_radial_bin_etas, dtype=np.int8)
    bs_radial_bin_p_values = np.zeros_like(bs_radial_bin_etas)
    for bs_idx in range(bootstrap_num):
        bs_etas = bs_radial_bin_etas[bs_idx]

        cdf_values = (bs_etas > bs_radial_bin_etas).sum(axis=0) / bootstrap_num
        greater_than = cdf_values > 1 - bernoulli_alpha / 2
        less_than = cdf_values < bernoulli_alpha / 2
        p_values = cdf_values.copy()
        p_values[greater_than] = 1 - cdf_values[greater_than]

        significances = np.zeros_like(p_values)
        significances[greater_than] = 1
        significances[less_than] = -1

        # Save results
        bs_radial_bin_p_values[bs_idx] = p_values
        bs_radial_bin_significances[bs_idx] = significances

    # Get bin centers
    radial_bin_centers = roi.recording['radial_bin_centers']

    # For original ETA
    vecs = _calc_preferred_directions(radial_bin_etas, radial_bin_significances > 0, radial_bin_centers)
    roi[f'preferred_vectors{postfix}'] = vecs

    # For Bootstrapped ETAs
    bs_vecs = np.zeros(bs_radial_bin_etas.shape[:2] + (2,))
    for bs_idx in range(bs_radial_bin_etas.shape[0]):
        bs_vecs[bs_idx] = _calc_preferred_directions(bs_radial_bin_etas[bs_idx],
                                                     bs_radial_bin_significances[bs_idx] > 0,
                                                     radial_bin_centers)

    # Get data
    closest_3_position_idcs = roi.recording['clostest_3_position_indices']

    # BS data

    # Trace clusters in original signal
    _, cluster_full_indices, unique_indices = create_clusters(radial_bin_significances > 0,
                                                              closest_3_position_idcs,
                                                              sign_radial_bin_threshold)
    roi[f'cluster_full_indices{postfix}'] = cluster_full_indices
    roi[f'cluster_unique_patch_indices{postfix}'] = unique_indices

    # Trace clusters in bootstrapped signals
    bs_cluster_full_indices = []
    bs_cluster_unique_indices = []
    for bs_idx in range(bootstrap_num):
        _, full_indices, unique_indices = create_clusters(bs_radial_bin_significances[bs_idx] > 0,
                                                          closest_3_position_idcs,
                                                          sign_radial_bin_threshold)
        bs_cluster_full_indices.append(full_indices)
        bs_cluster_unique_indices.append(unique_indices)
    roi[f'bs_cluster_full_indices{postfix}'] = bs_cluster_full_indices
    roi[f'bs_cluster_unique_patch_indices{postfix}'] = bs_cluster_unique_indices

    # Calculate cluster significances (2nd bootstrap)

    bs_cluster_full_indices = roi[f'bs_cluster_full_indices{postfix}']

    # Convert to float
    radial_bin_significances_float = (radial_bin_significances > 0).astype(np.float64)
    bs_radial_bin_significances_float = (bs_radial_bin_significances > 0).astype(np.float64)

    # For bootstrapped clusters
    bs_max_cluster_scores = np.zeros(bootstrap_num)
    for bs_idx in range(bootstrap_num):
        bs_indices = bs_cluster_full_indices[bs_idx]
        bs_significances = bs_radial_bin_significances_float[bs_idx]

        # Calculate largest sum of scores for all clusters
        _scores = [bs_significances[tuple(_idcs.T)].sum() for _idcs in bs_indices]
        bs_max_cluster_scores[bs_idx] = np.max(_scores) if len(_scores) > 0 else 0

    # Check significance of cluster scores in original ETA
    _scores = [radial_bin_significances_float[tuple(_idcs.T)].sum() for _idcs in cluster_full_indices]
    original_cluster_scores = np.array(_scores)

    # counts, bins, patches = plt.hist(bs_max_cluster_scores, bins=50)
    # plt.vlines(original_cluster_scores, 0, counts.max(), color='red')
    # plt.show()

    cluster_significances = (original_cluster_scores >= bs_max_cluster_scores[:, None]).sum(axis=0) / bootstrap_num > (1 - cluster_alpha)
    cluster_significant_indices = np.where(cluster_significances)[0]

    # Save results
    roi[f'cluster_significant_indices{postfix}'] = cluster_significant_indices
    roi[f'has_receptive_field{postfix}'] = len(cluster_significant_indices) > 0


def _calc_preferred_directions(bin_etas: np.ndarray, bin_significances: np.ndarray,
                               bin_centers: np.ndarray) -> np.ndarray:
    """
    bin_etas: shape (patch_num, bin_num)
    bin_significances: shape (patch_num, bin_num)
    bin_centers: shape (bin_num,)
    """

    # Calculate direction vectors for given angles
    direction_vectors = np.array([[np.cos(a), np.sin(a)] for a in bin_centers])

    # Calculate population vector for each patch based on significant direction bins
    population_vectors = np.zeros(bin_etas.shape[:1] + (2,))
    for idx, (etas, signs) in enumerate(zip(bin_etas, bin_significances)):

        if np.any(signs):

            # Select excitatory bins
            idcs = np.where(signs)[0]

            # Calculate
            vecs = etas[idcs][:, None] * direction_vectors[idcs]
            vec_pop = np.sum(vecs, axis=0) / np.sum(etas)

        else:
            vec_pop = np.array([0, 0])

        # Append
        population_vectors[idx] = vec_pop

    return population_vectors


def calculate_egomotion_similarities(roi: Roi, postfix: str = ''):
    """"""

    if not roi[f'has_receptive_field{postfix}']:
        return

    # Get recording data
    positions = roi.recording['positions']

    # Select all patches that belong to significant clusters
    significant_cluster_indices = roi[f'cluster_significant_indices{postfix}']
    cluster_unique_patch_indices = roi[f'cluster_unique_patch_indices{postfix}']
    significant_clusters = [cluster_unique_patch_indices[_idx] for _idx in significant_cluster_indices]

    significant_patch_indices = [x for y in significant_clusters for x in y]
    roi[f'significant_patch_indices{postfix}'] = significant_patch_indices
    # Select preferred local directions
    preferred_vectors = roi[f'preferred_vectors{postfix}']
    preferred_velocities = np.linalg.norm(preferred_vectors, axis=1)

    rf_selection = np.zeros_like(preferred_velocities, dtype=bool)
    rf_selection[significant_patch_indices] = True

    if sum(rf_selection) == 0:
        return

    # Create set of motion axes
    egomotion_axes_opts = roi.recording['egomotion_axes_opts']

    # Calculate similarities to translation templates
    egomotion_translation_local_motion = roi.recording['egomotion_translation_local_motion']
    translation_similarities_all = np.zeros((egomotion_axes_opts.shape[0], rf_selection.sum()))
    for i in range(translation_similarities_all.shape[0]):
        for j in range(translation_similarities_all.shape[1]):
            v = egomotion_translation_local_motion[i][rf_selection][j]
            v_pref = preferred_vectors[rf_selection][j]
            translation_similarities_all[i, j] = np.dot(v, v_pref) / (np.linalg.norm(v) * np.linalg.norm(v_pref))
    translation_similarities = translation_similarities_all.mean(axis=1)

    # Save to file
    roi[f'translation_similarities_all{postfix}'] = translation_similarities_all
    roi[f'translation_similarities{postfix}'] = translation_similarities
    selected_translation_idx = np.argmax(translation_similarities)
    best_translation_similarity = translation_similarities[selected_translation_idx]
    roi[f'translation_best_similarity{postfix}'] = best_translation_similarity
    roi[f'translation_vectors{postfix}'] = egomotion_translation_local_motion[selected_translation_idx]
    roi[f'translation_axis{postfix}'] = egomotion_axes_opts[selected_translation_idx]

    # Calculate similarities to rotation templates
    egomotion_rotation_local_motion = roi.recording['egomotion_rotation_local_motion']
    rotation_similarities_all = np.zeros((egomotion_axes_opts.shape[0], rf_selection.sum()))
    for i in range(rotation_similarities_all.shape[0]):
        for j in range(rotation_similarities_all.shape[1]):
            v = egomotion_rotation_local_motion[i][rf_selection][j]
            v_pref = preferred_vectors[rf_selection][j]
            rotation_similarities_all[i, j] = np.dot(v, v_pref) / (np.linalg.norm(v) * np.linalg.norm(v_pref))
    rotation_similarities = rotation_similarities_all.mean(axis=1)

    # Save to file
    roi[f'rotation_similarities{postfix}'] = rotation_similarities
    selected_rotation_idx = np.argmax(rotation_similarities)
    best_rotation_similarity = rotation_similarities[selected_rotation_idx]
    roi[f'rotation_best_similarity{postfix}'] = best_rotation_similarity
    roi[f'rotation_vectors{postfix}'] = egomotion_rotation_local_motion[selected_rotation_idx]
    roi[f'rotation_axis{postfix}'] = egomotion_axes_opts[selected_rotation_idx]

    roi[f'is_translation_selective{postfix}'] = best_translation_similarity > best_rotation_similarity
    roi[f'is_rotation_selective{postfix}'] = best_rotation_similarity > best_translation_similarity


def project_to_local_2d_vectors(normals: np.ndarray, vectors: np.ndarray,
                                vertical_up_direction: Union[List, Tuple, np.ndarray] = None) -> np.ndarray:

    if vertical_up_direction is None:
        vertical_up_direction = [0, 0, 1]

    if not isinstance(vertical_up_direction, np.ndarray):
        vertical_up_direction = np.array(vertical_up_direction)

    vnorms = vertical_up_direction - normals * np.dot(normals, np.array([0, 0, 1]))[:, None]
    vnorms /= np.linalg.norm(vnorms, axis=1)[:, None]

    hnorms = -crossproduct(vnorms, normals)
    hnorms /= np.linalg.norm(hnorms, axis=1)[:, None]

    vectors_2d = np.zeros((*vectors.shape[:2], 2))
    for i, v in enumerate(vectors):
        # Calculate 2d motion vectors in coordinate system defined by local horizontal and vertical norms
        motvecs_2d = np.array([np.sum(v * hnorms, axis=1),
                               np.sum(v * vnorms, axis=1)])
        vectors_2d[i] = motvecs_2d.T

    return vectors_2d


def calculate_local_directions(motvecs: np.ndarray, bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Convert to angle and velocity
    motion_angles = np.arctan2(motvecs[:, :, 1], motvecs[:, :, 0])
    motion_velocities = np.linalg.norm(motvecs, axis=2)
    # Backwards transform (keep for reference):
    # motion_vectors_2d = np.zeros((*motion_angles.shape[:2], 2))
    # motion_vectors_2d[:,:,0] = motion_velocities * np.cos(motion_angles)
    # motion_vectors_2d[:,:,1] = motion_velocities * np.sin(motion_angles)

    # Calculate bin vectors for each patch and frame weighted by the local velocities
    bin_norms = motion_velocities[:, :, None] * np.logical_and(bin_edges[:-1] <= motion_angles[:, :, None],
                                                               motion_angles[:, :, None] <= bin_edges[1:])

    # Calculate ETAs
    bin_etas = np.mean(bin_norms, axis=0)

    return bin_norms, bin_etas


def calculate_local_directions_torch(motvecs: torch.Tensor, bin_edges: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate local directions and ETAs for motion vectors using PyTorch.

    Args:
        motvecs (torch.Tensor): Motion vectors with shape (M, N, 2).
        bin_edges (torch.Tensor): Bin edges with shape (K,).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Bin norms and ETAs.
    """
    # Convert to angle and velocity
    motion_angles = torch.atan2(motvecs[:, :, 1], motvecs[:, :, 0])
    motion_velocities = torch.norm(motvecs, dim=2)

    # Calculate bin vectors for each patch and frame
    bin_norms = (motion_velocities[:, :, None]
                 * ((bin_edges[:-1] <= motion_angles[:, :, None])
                    & (motion_angles[:, :, None] < bin_edges[1:])).float())

    # Calculate ETAs
    bin_etas = torch.mean(bin_norms, 0)

    return bin_norms, bin_etas


def crossproduct(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Workaround because of NoReturn situation in numpy.cross"""

    return np.cross(v1, v2)


def create_clusters(significant_bins: np.ndarray[bool],
                    closest_indices: np.ndarray[int],
                    sign_radial_bin_threshold: int) -> Tuple[np.ndarray, list, list]:
    """
    significant_bins: bool array with shape (patches x radial bins)
    closest_indices: int array with shape (patches x 3) containing indices of the three closest patches for each patch
    sign_radial_bin_threshold: int threshold for how many significant bins adjacent neighbors need to have in common
        in order to be considered "connected"

    returns
        np.ndarray: shape (patches x patches x radial bins) of bool values, marking valid parts of the cluster
    """

    # Build cluster_idcs starting at each individual patch
    patch_num, radial_bin_num = significant_bins.shape
    cluster_maps = np.zeros((patch_num, patch_num, radial_bin_num), dtype=bool)
    visited_patch_indices = []
    for patch_start_idx in range(significant_bins.shape[0]):
        # print(f'Start at patch {patch_start_idx}')

        # Skip patch, if it is already part of another cluster
        if patch_start_idx in visited_patch_indices:
            continue

        _trace_cluster(patch_start_idx,
                       significant_bins,
                       closest_indices,
                       sign_radial_bin_threshold,
                       cluster_maps[patch_start_idx],
                       visited_patch_indices)

    # Get [patch, bin] indices for all possible cluster maps
    cluster_indices = [np.argwhere(_map) for _map in cluster_maps if _map.sum() > 0]

    # Get unique patch indices
    unique_patch_indices = list({tuple(np.unique(_idcs[:, 0])): None for _idcs in cluster_indices}.keys())

    return cluster_maps, cluster_indices, unique_patch_indices


def _trace_cluster(current_patch_idx: int,
                   significant_bins: np.ndarray[bool],
                   closest_indices: np.ndarray[int],
                   sign_radial_bin_threshold: int,
                   cluster_map: np.ndarray[bool],
                   visited_patch_indices: List[int]):
    """Recursively go through all spatially connected patches
    which share the same significant direction bins
    """

    # Select bins
    current_sign_bins = significant_bins[current_patch_idx]

    # Check if enough significant bins in current patch
    if not sum(current_sign_bins) >= sign_radial_bin_threshold:
        return

    # print(f'> {current_patch_idx}')
    # Add patch data
    visited_patch_indices.append(current_patch_idx)

    # Go through adjacent patches
    for new_patch_idx in closest_indices[current_patch_idx]:

        new_sign_bins = significant_bins[new_patch_idx]
        sign_bins_in_common = current_sign_bins & new_sign_bins

        # At least <sign_radial_bin_threshold> significant direction bins in common
        if not (sum(sign_bins_in_common) >= sign_radial_bin_threshold) \
                or (new_patch_idx in visited_patch_indices):
            continue

        # Set cluster map bins for matching significant bins with this neighbor to true
        cluster_map[current_patch_idx, sign_bins_in_common] = True
        cluster_map[new_patch_idx, sign_bins_in_common] = True

        # Continue tracing for next patch
        _trace_cluster(new_patch_idx,
                       significant_bins,
                       closest_indices,
                       sign_radial_bin_threshold,
                       cluster_map,
                       visited_patch_indices)
