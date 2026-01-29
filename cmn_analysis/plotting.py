from typing import List

import matplotlib
import numpy as np
import quaternionic
from matplotlib import animation, pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
import scipy

import caload
from caload.schemata.ca_imaging_s2p_vxpy import *
import helper
from functions import project_to_local_2d_vectors

colors_inh_exc = [(0 / 255, 0 / 255, 255 / 255), (150 / 255, 150 / 255, 150 / 255), (255 / 255, 0 / 255, 0 / 255)]
nodes_inh_exc = [0.0, 0.5, 1.0]
significance_cmap = LinearSegmentedColormap.from_list('blue_gray__red', list(zip(nodes_inh_exc, colors_inh_exc)))


def plot_dffs(rois: caload.entities.EntityCollection,
              highlight: str = None, classification: bool = False, yscale: float = 1 / 4, yoffset = 0):
    highlight_values = None
    if highlight is not None:
        highlight_values = [roi[highlight] for roi in rois]
        corrmax = max(highlight_values)
        colors = np.array(highlight_values)
        colors /= corrmax
        colors[colors < 0] = 0
        colors = 1 - colors
    else:
        colors = np.zeros(len(rois))

    gs = plt.GridSpec(1, 10)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(gs[:, 2:-1])

    for roi in list(rois):
        print(f'Plot {roi}')
        # times = roi.recording['ca_times']
        # dff = roi['dff'] * yscale
        times = roi.recording['time_resampled']
        dff = roi['dff_resampled']
        times -= times[0]
        ax.plot(times, yoffset + dff, color=np.ones(3) * colors[yoffset], linewidth=0.8)
        ax.annotate(str(roi), (-10, yoffset + 0.5), verticalalignment='center', horizontalalignment='right',
                    fontsize='xx-small')
        if highlight_values is not None:
            ax.annotate(f'{highlight_values[yoffset]:.3f}',
                        (times[-1] + 10, yoffset + 0.5),
                        verticalalignment='center', horizontalalignment='left', fontsize='xx-small')

        if classification & roi['signal_length'] > 0:
            dff_masked = dff.copy()
            dff_masked[~roi['signal_selection']] = np.nan
            ax.plot(times, yoffset + dff_masked, color='red', linewidth=0.8)
        yoffset += 1

    ax.axis('off')
    ax.set_xlim(-100, ax.get_xlim()[1])
    # fig.tight_layout()
    plt.show()


def plot_distribution(entities: caload.entities.EntityCollection, *attributes,
                      ax: plt.Axes = None, **kwargs):
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(1, 1)

    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.5 if len(attributes) > 1 else 1.0

    for attr in attributes:
        values = entities[attr]
        ax.hist(values, label=attr, **kwargs)

    if show:
        plt.show()


def plot_event_classification_debug(roi: Roi):
    # Recording
    cmn_selection = roi.recording['cmn_phase_selection_original']
    times = roi.recording['ca_times']
    times[~cmn_selection] = np.nan
    # Roi
    dff = roi['dff']
    dffcmn = dff[cmn_selection]
    bestidx = roi['best_dff_threshold_index']
    bestthresh = roi['best_dff_threshold']
    signal_classification = roi['signal_classification']

    thresholds = signal_classification['thresholds']
    rvals = signal_classification['rvals']
    pvals = signal_classification['pvals']
    slopes = signal_classification['slopes']
    intercepts = signal_classification['intercepts']
    bin_num = signal_classification['bin_num']

    basedff = dffcmn[dffcmn <= bestthresh]
    signaldff = dffcmn[dffcmn > bestthresh]

    fig = plt.figure(figsize=(10, 4))
    gs = plt.GridSpec(3, 3)
    axes = [fig.add_subplot(gs[:2, i]) for i in range(3)]
    axdff = fig.add_subplot(gs[2, :])
    axes[1].set_title(roi)

    bins = np.linspace(dffcmn.min(), dffcmn.max(), bin_num)
    bin_width = bins[1] - bins[0]
    bin_centers = bins[:-1] + (bins[1] - bins[0]) / 2
    basecounts, _, _ = axes[0].hist(basedff, color='tab:blue', alpha=0.3, bins=bins, label='baseline')
    if len(signaldff) > 0:
        signalcounts, _, _ = axes[0].hist(signaldff, color='tab:orange', alpha=0.3, bins=bins, label='signal')
    else:
        signalcounts = np.nan * np.ones(len(bin_centers))
    pdf = scipy.stats.norm.pdf(bin_centers, loc=np.mean(basedff), scale=np.std(basedff))
    pdf_counts = pdf * len(basedff) * bin_width
    axes[0].plot(bin_centers, pdf_counts, color='black')
    axes[0].set_xlabel('DFF')
    axes[0].set_ylabel('#')

    # Plot rvals and pvals
    axes[1].plot(thresholds, rvals, color='tab:blue')
    axes[1].plot([bestthresh], [rvals[bestidx]], 'o', color='tab:red')
    axes[1].set_xlabel('Threshold [DFF]')
    axes[1].set_ylabel('R')

    axes[2].scatter(basecounts, pdf_counts, color='tab:blue', s=10, label=f'baseline ({len(basedff)})')
    axes[2].scatter(signalcounts, pdf_counts, s=10, color='tab:orange', label=f'signal ({len(signaldff)})')
    axes[2].plot([basecounts.min(), basecounts.max()],
                 [basecounts.min() * slopes[bestidx] + intercepts[bestidx],
                  basecounts.max() * slopes[bestidx] + intercepts[bestidx]],
                 label=f'R: {rvals[bestidx]:.5f}, p: {pvals[bestidx]:.5f}')

    axes[2].legend(prop={'size': 6})
    axes[2].set_xlabel('observed #')
    axes[2].set_ylabel('normpdf #')
    plt.tight_layout()

    # Plot signal
    axdff.plot(times, dff, color='black')
    dffsignalout_thresh = np.nan * np.ones_like(dff)
    signal_selection_thresh = cmn_selection & (dff > bestthresh)
    dffsignalout_thresh[signal_selection_thresh] = signaldff

    # Get transition points between baseline/signal (True/False)
    signal_selection = np.copy(signal_selection_thresh)
    diff_array = np.diff(signal_selection_thresh.astype(np.float64), prepend=[0])
    segm_start_idcs = np.where(diff_array > 0)[0]
    segm_end_idcs = np.where(diff_array < 0)[0]
    # Truncate start index array if last segment end is missing
    if segm_end_idcs.shape[0] < segm_start_idcs.shape[0]:
        segm_start_idcs = segm_start_idcs[:-1]
    # Remove any signal after local peak within segment
    for i1, i2 in zip(segm_start_idcs, segm_end_idcs):
        idx_max = np.argmax(dff[i1:i2]) + 1
        # Assume that actual event started ~one frame before we detect it because of slow sampling
        signal_selection[i1 - 1] = True
        signal_selection[i1 + idx_max:i2] = False

    dffsignalout = dff.copy()
    dffsignalout[~signal_selection] = np.nan
    axdff.plot(times, dffsignalout, '-', color='tab:green')
    # Plot dots on top
    axdff.plot(times, dffsignalout_thresh, '.', markersize=3., color='tab:orange')


def plot_classifications(rois: caload.entities.EntityCollection):
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    i = 0
    for roi in rois:
        plot_classification(ax, roi, yoffset=i, yscale=1 / 4)
        i += 1

    fig.tight_layout()
    plt.show()


def plot_classification(ax: plt.Axes, roi: Roi, yoffset: float = 0., yscale: float = 1., postfix: str = ''):
    # times = roi.recording['ca_times']
    # dff = roi['dff'] * yscale
    times = roi.recording['time_resampled']
    dff = roi['dff_resampled'] * yscale
    signal_selection = roi[f'signal_selection{postfix}']

    # Plot
    ax.plot(times, yoffset + dff, color='black', linewidth=0.8)
    dff_masked = dff.copy()
    dff_masked[~signal_selection] = np.nan
    ax.plot(times, yoffset + dff_masked, color='red', linewidth=1.)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('DFF')


def plot_translation_quiver_2d(direction: np.ndarray, normals: np.ndarray, ax: plt.Axes):
    # Just stack for each face and add new 1st dim for compat. with project_to_local_2d_vectors function
    direction_vecs = np.repeat(direction[None, :], normals.shape[0], 0)[None, :, :]
    azimuths, elevations, _ = helper.cart2sph(*normals.T)
    direction_vecs_2d = project_to_local_2d_vectors(normals, direction_vecs).squeeze()

    ax.quiver(azimuths, elevations, *direction_vecs_2d.T,
              color='green', pivot='tail', units='width', width=0.002, scale=50, alpha=0.4)


def plot_radial_histograms(ax: plt.Axes,
                           positions: np.ndarray, lengths: np.ndarray, bin_edges: np.ndarray,
                           scale: float = 1.0) -> List[PatchCollection]:
    # normlengths = (lengths / lengths.max()) ** 2
    normlengths = lengths / lengths.max()
    normlengths = normlengths ** 2
    normlengths *= 1 / 10
    for pos, lens in zip(positions, normlengths):
        # Calculate spherical coordinates
        az, el, _ = helper.cart2sph(*pos)

        # Create radial histogram patches
        coll = PatchCollection([Polygon(
            np.array([[az, el],
                      [az + _len * scale * np.cos(edge1), el + _len * scale * np.sin(edge1)],
                      [az + _len * scale * np.cos(edge2), el + _len * scale * np.sin(edge2)],
                      [az, el],
                      ])) for _len, edge1, edge2 in zip(lens, bin_edges[:-1], bin_edges[1:])],
            color='gray', edgecolor='black', linewidth=.1)

        # Add to axes
        ax.add_collection(coll)

    xticks = np.linspace(-np.pi, np.pi, 5)
    ax.set_xticks(xticks, [int(v / np.pi * 180) for v in xticks])
    ax.set_xlim(-np.pi - 0.2, np.pi + 0.2)
    ax.set_xlim(-np.pi - 0.2, np.pi + 0.2)
    ax.set_xlabel('azimuth [deg]')
    yticks = np.linspace(-np.pi / 2, np.pi / 2, 5)
    ax.set_yticks(yticks, [int(v / np.pi * 180) for v in yticks])
    ax.set_ylim(-np.pi / 2 + np.radians(1), np.pi / 2 - np.radians(1))
    ax.set_ylabel('elevation [deg]')
    ax.set_aspect('equal')

    return ax._children


def add_grid_patch_coordinates(_vertices: np.ndarray, _patches, duplicated, projection: str = None):
    p = []
    center = _vertices.mean(axis=0)
    caz, cel, _ = helper.cart2sph(*center)
    for j in range(3):
        # Get vertices
        v1 = _vertices[j]
        v2 = _vertices[j + 1]

        # Convert
        q1 = quaternionic.array([0, *v1])
        q2 = quaternionic.array([0, *v2])

        for k in np.linspace(0, 1, 20):

            # Do spherical interpolation
            vinterp = quaternionic.slerp(q1, q2, k).ndarray[1:]
            az, el, _ = helper.cart2sph(*vinterp)

            # If there is a sign flip at the back around +/-180deg,
            #  duplicate patch once for an ever so slightly rotated version
            #  and correct azimuth on this patch
            if (np.sign(az) != np.sign(caz)) & (np.abs(caz) > np.pi/2):

                # Duplicate patch on opposite side
                if not duplicated:
                    yaw_rot = np.sign(caz)/1000
                    # M = transforms.rotate(np.sign(caz)/1000, (0, 0, 1))[:3, :3]
                    M = np.array([[np.cos(yaw_rot), -np.sin(yaw_rot), 0],
                                  [np.sin(yaw_rot), np.cos(yaw_rot), 0],
                                  [0, 0, 1]])
                    add_grid_patch_coordinates(np.array([np.dot(v, M) for v in _vertices]), _patches, True)
                    duplicated = True

                # Set interpolated azimuth position to correct side
                if np.sign(caz) < 0:
                    az = np.deg2rad(-180)
                else:
                    az = np.deg2rad(180)

            # Apply projection
            if projection is None:
                x = az
                y = el
            elif projection == 'mollweide':
                x, y = helper.mollweide_projection(az, el)
            elif projection == 'eckert_iv':
                x, y = helper.eckert_iv_projection(az, el)
            else:
                raise Exception('No valid projection provided')

            # Add coordinate for this patch
            p.append([x, y])

    # Add patch coordinates to global list
    _patches.append(p)


def plot_patch_grid(ax: plt.Axes, recording: Recording, **props) -> PatchCollection:

    settings = dict(color='None', edgecolor='black', linewidth=.5)
    settings.update(props)

    corners = recording['patch_corners']
    indices = recording['patch_indices']
    x = []
    y = []
    patches = []
    # Iterate through patches
    for i in range(indices.shape[0])[::3]:
        vertices = np.append(corners[i:i + 3], corners[None, i], axis=0)

        # Calculate spherical coordinates
        azims, elevs, _ = helper.cart2sph(*vertices.T)

        add_grid_patch_coordinates(vertices, patches, False)

    ax.scatter(x, y, s=10, c='black')

    # Create grid patches
    coll = PatchCollection([Polygon(np.array(p)) for p in patches], **settings)
    ax.add_collection(coll)

    return coll


def plot_patches(ax: plt.Axes, roi: Roi, patch_indices: np.ndarray, **kwargs):
    corners = roi.recording['patch_corners']
    patches = []
    # Iterate through patches
    for i in patch_indices:
        vertices = np.append(corners[i * 3:(i + 1) * 3], corners[None, i * 3], axis=0)

        # Calculate spherical coordinates
        azims, elevs, _ = helper.cart2sph(*vertices.T)

        add_grid_patch_coordinates(vertices, patches, False)

    # Create grid patches
    coll = PatchCollection([Polygon(np.array(p)) for p in patches], **kwargs)
    ax.add_collection(coll)

    return coll


def plot_radial_significance(roi: Roi, patch_collections: List[PatchCollection], postfix: str = ''):

    # Get bin significances
    radial_bin_significance = roi[f'radial_bin_significances{postfix}']

    # Mark significance
    for i, coll in enumerate(patch_collections):
        coll.set_cmap(significance_cmap)
        coll.set_clim(-1, 1)
        coll.set_array(radial_bin_significance[i])


def plot_rf_overview(roi: Roi, save_path: str = None, postfix: str = ''):

    # ROI data
    radial_bin_etas = roi[f'radial_bin_etas{postfix}']

    # Recording data
    radial_bin_edges = roi.recording['radial_bin_edges']
    positions = roi.recording['positions']

    namestr  = f'{str(roi)}{postfix}'

    # Plot DFF
    gs = plt.GridSpec(12, 6)
    fig = plt.figure(figsize=(14, 10), num=namestr)
    ax_dff = fig.add_subplot(gs[:2, :])
    ax_dff.set_title(str(roi))
    plot_classification(ax_dff, roi, postfix=postfix)

    # Plot radial histogram figure
    ax_hist = fig.add_subplot(gs[2:7, 2:])
    patch_collections = plot_radial_histograms(ax_hist, positions, radial_bin_etas, radial_bin_edges)
    plot_radial_significance(roi, patch_collections, postfix=postfix)
    plot_patch_grid(ax_hist, roi.recording)

    # Plot preferred local motion vectors quiver plot
    ax_quiv = fig.add_subplot(gs[7:, 2:], sharex=ax_hist, sharey=ax_hist)

    # If Roi has a recepotive field, plot best egomotion fit
    if roi[f'has_receptive_field{postfix}']:
        # Get position and local motion preference data
        x, y, _ = helper.cart2sph(*positions.T)
        preferred_vectors = roi[f'preferred_vectors{postfix}']
        preferred_velocities = np.linalg.norm(preferred_vectors, axis=1)
        if roi[f'translation_best_similarity{postfix}'] > roi[f'rotation_best_similarity{postfix}']:
            translation_vecs = roi[f'translation_vectors{postfix}']
            ax_quiv.quiver(x, y, translation_vecs[:, 0], translation_vecs[:, 1],
                       pivot='mid', color='black', alpha=0.3, width=0.002, scale=30)
        else:
            rotation_vecs = roi[f'rotation_vectors{postfix}']
            ax_quiv.quiver(x, y, rotation_vecs[:, 0], rotation_vecs[:, 1],
                       pivot='mid', color='black', alpha=0.3, width=0.002, scale=30)

        significant_cluster_indices = roi[f'cluster_significant_indices{postfix}']
        cluster_unique_patch_indices = roi[f'cluster_unique_patch_indices{postfix}']
        selected_clusters = [cluster_unique_patch_indices[_idx] for _idx in significant_cluster_indices]
        for i, idcs in enumerate(selected_clusters):
            idcs = np.array(idcs)
            color = matplotlib.colormaps['tab10'](i)
            ax_quiv.quiver(x[idcs], y[idcs], preferred_vectors[idcs, 0], preferred_vectors[idcs, 1],
                       pivot='mid', color=color, width=0.002, scale=preferred_velocities[idcs].max() * 30)

            # for xx, yy, vel, px, py in zip(x[idcs], y[idcs], pref_velocities[idcs], pref_x[idcs], pref_y[idcs]):
            #     ax3.text(xx, yy, f'{vel:.2f}')
            # ax2.text(xx, yy, f'{vel:.2f}\n{px:.2f}/{py:.2f}')

    # Add grid
    plot_patch_grid(ax_quiv, roi.recording)

    # Format
    ax_quiv.set_xlabel('azimuth [deg]')
    ax_quiv.set_ylabel('elevation [deg]')
    ax_quiv.set_aspect('equal')

    # # Plot cluster scores and classification
    # cluster_scores = roi['cluster_scores']
    # bs_cluster_scores = roi['bs_cluster_scores']
    # cluster_significances = roi['cluster_significances']
    # cluster_gmm = roi['cluster_gmm']
    #
    # ax_gmm = fig.add_subplot(gs[2:6, :2])
    #
    # rv = scipy.stats.multivariate_normal(mean=cluster_gmm.means_[0],
    #                                      cov=cluster_gmm.covariances_[0])
    #
    # ax_gmm.scatter(*bs_cluster_scores, s=2, alpha=1., c='gray')
    # ax_gmm.scatter(*cluster_scores[:, ~cluster_significances], s=5, alpha=1, c='blue')
    # ax_gmm.scatter(*cluster_scores[:, cluster_significances], s=5, alpha=1., c='red')
    # ax_gmm.set_title('Cluster scores')
    # ax_gmm.set_xlabel('Cluster max. p-scores')
    # ax_gmm.set_ylabel('Sim. to patches')
    # x, y = np.meshgrid(np.linspace(*ax_gmm.get_xlim(), 1000),
    #                    np.linspace(*ax_gmm.get_ylim(), 1000))
    # pos = np.dstack((x, y))
    # ax_gmm.contourf(x, y, rv.pdf(pos), levels=100, alpha=0.2)

    # Plot egomotion similarities
    if 'egomotion_axes_opts' in roi and roi[f'has_receptive_field{postfix}']:
        ax_trans_sim, ax_rot_sim = fig.add_subplot(gs[6:8, :1]), fig.add_subplot(gs[6:8, 1:2])
        egomotion_axes = roi['egomotion_axes_opts']
        axis_az, axis_el, _ = helper.cart2sph(*egomotion_axes.T)
        ax_trans_sim.scatter(axis_az, axis_el, s=10, c=roi[f'translation_similarities{postfix}'])
        ax_trans_sim.set_title('trans. similarity')
        ax_trans_sim.set_xlabel('azimuth [rad]')
        ax_trans_sim.set_ylabel('elevation [rad]')
        ax_rot_sim.scatter(axis_az, axis_el, s=10, c=roi[f'rotation_similarities{postfix}'])
        ax_rot_sim.set_title('rot. similarity')
        ax_rot_sim.set_xlabel('azimuth [rad]')

    # Format
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(f'{save_path}/{namestr}.png', dpi=300)
        fig.savefig(f'{save_path}/{namestr}.pdf', dpi=300)

        plt.close(fig)
    else:
        plt.show()


def animate_motion_matrix(times: np.ndarray, positions: np.ndarray, motion_vectors, sample_rate):
    print('Create vector field animation')

    # Calculate spherical coordinates
    azims, elevs, _ = helper.cart2sph(*positions.T)

    # Calculate 2D motion vectors
    motion_vectors_2d = project_to_local_2d_vectors(positions, motion_vectors)
    # Normalize to control arrow length for visualization
    motion_vectors_2d /= motion_vectors_2d.max()

    # Create figure
    fig = plt.figure(figsize=(5, 2.5), dpi=200)
    ax = plt.subplot()
    ax.set_aspect(1)
    qr = ax.quiver(azims, elevs, 0, 0, pivot='middle', scale=25, width=0.0015, headlength=4.5)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], [-180, -90, 0, 90, 180])
    ax.set_xticklabels()
    ax.set_xlabel('azimuth [deg]')
    ax.set_yticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2], [-90, -45, 0, 45, 90])
    ax.set_ylabel('elevation [deg]')
    fig.tight_layout()

    # Create animation
    def animate(tidx, qr):
        qr.set_UVC(*motion_vectors_2d[tidx].T)

        return (qr,)

    ani = animation.FuncAnimation(fig, animate,
                                  fargs=(qr,), interval=1000 / sample_rate,
                                  blit=True, frames=times.shape[0])

    plt.show()
    # ani.save(f'./motion_vectors_f{frame_num}_sp{sp_sigma}_tp{tp_sigma}.mp4', writer='ffmpeg')
