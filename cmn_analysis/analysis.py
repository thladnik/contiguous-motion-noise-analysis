import numpy as np
import pandas as pd
import tifffile

import caload
from caload.schemata.ca_imaging_s2p_vxpy import *

from . import functions
from . import helper


def create(analysis_path: str, recreate: bool = False, **kwargs):

    # Delete analysis
    if recreate:
        caload.analysis.delete_analysis(analysis_path)

    # Create analysis
    dbhost = kwargs.get('dbhost')
    dbname = kwargs.get('dbname')
    dbuser = kwargs.get('dbuser')
    dbpassword = kwargs.get('dbpassword')

    caload.analysis.create_analysis(analysis_path=analysis_path, entity_schema=schema,
                                    dbhost=dbhost, dbname=dbname, dbuser=dbuser, dbpassword=dbpassword)


def digest_new_data(analysis_path: str, data_root_path: str, **kwargs):

    # Open and digest
    analysis = caload.analysis.open_analysis(analysis_path)

    sync_type = kwargs.get('sync_type')
    sync_signal = kwargs.get('sync_signal')
    sync_signal_time = kwargs.get('sync_signal_time')
    frame_avg_num = kwargs.get('frame_avg_num', 1)
    caload.analysis.digest_data(analysis, digest, data_root_path,
                                sync_type=sync_type, sync_signal=sync_signal, sync_signal_time=sync_signal_time,
                                frame_avg_num=frame_avg_num)


def run(analysis_path: str):

    analysis = caload.analysis.open_analysis(analysis_path)

    # Process recordings
    recordings = analysis.get(Recording, 'NOT(EXIST(time_resampled))')

    recordings.map_async(functions.process_recording, sample_rate=10, radial_bin_num=16)

    # Process ROIs
    (analysis.get(Roi, 'NOT(EXIST(dff))')
     .map_async(functions.calculate_dff, window_size=120, percentile=10))

    (analysis.get(Roi, 'NOT(EXIST(rotation_autocorrelation))')
     .map_async(functions.calculate_autocorrelations))

    (analysis.get(Roi, 'NOT(EXIST(signal_dff_mean))')
     .map_async(functions.detect_events_with_derivative))

    (analysis.get(Roi, 'NOT(EXIST(bs_radial_bin_etas))')
     .map_async(functions.calculate_reverse_correlations, worker_num=10, bootstrap_num=1000, use_gpu=True))

    (analysis.get(Roi).where('NOT(EXIST(has_receptive_field))')
     .map_async(functions.run_2step_ncb_test, bernoulli_alpha=0.05))

    # Select all ROIs that have RF and calculate similarity to egomotion templates
    (analysis.get(Roi, '(has_receptive_field == True) AND NOT(EXIST(is_rotation_selective))')
     .map_async(functions.calculate_egomotion_similarities))


def postprocessing(analysis_path: str):

    analysis = caload.analysis.open_analysis(analysis_path)

    cmn_rois = analysis.get(Roi, 'has_receptive_field == True')
    df = cmn_rois.dataframe

    # Calculate indices
    df['motion_selectivity_index'] = (
            (df['translation_best_similarity'] - df['rotation_best_similarity']) / (
            df['translation_best_similarity'] + df['rotation_best_similarity']))

    # Calculate other stuff
    df['1_minus_rotation_similarity'] = 1 - df['rotation_best_similarity']
    df['1_minus_translation_similarity'] = 1 - df['translation_best_similarity']

    df['cluster_significant_num'] = df['cluster_significant_indices'].apply(lambda x: len(x))

    df.commit()

    # Add convencience access to properties
    def _unpack_indices(series: pd.Series):
        _patch_idcs = []
        for _cluster_idx in series['cluster_significant_indices']:
            _patch_idcs.extend(series['cluster_unique_patch_indices'][_cluster_idx])
        return _patch_idcs

    sub_df = df['cluster_significant_indices', 'cluster_unique_patch_indices']

    rf_significant_patch_indices = sub_df.apply(_unpack_indices, axis=1)
    df['rf_significant_patch_num'] = [len(idcs) for idcs in rf_significant_patch_indices]

    # Convert patch number to approx estimate of area
    patch_centers = cmn_rois[0].recording['positions']
    patch_num = patch_centers.shape[0]
    df['rf_size_sr'] = df['rf_significant_patch_num'] * 4 * np.pi / patch_num

    # Calculate response type based on mode number
    df['roi_response_type'] = df['cluster_significant_num'].apply(lambda x: 'simple' if x == 1 else 'complex')

    df.commit()

    # Calculate egomotion axes for all
    df[f'translation_axis_sph'] = df[f'translation_axis'].apply(lambda v: helper.cart2sph(*v))
    df[f'translation_azimuth'], df[f'translation_elevation'], _ = np.stack(df[f'translation_axis_sph'].values).T

    df[f'rotation_axis_sph'] = df[f'rotation_axis'].apply(lambda v: helper.cart2sph(*v))
    df[f'rotation_azimuth'], df[f'rotation_elevation'], _ = np.stack(df[f'rotation_axis_sph'].values).T

    df.commit()


def classify_brain_area(rois: RoiCollection, region_map: str, area_name: dict):

    def coordinates_in(sub_df: pd.DataFrame, voxel_volume: np.ndarray):
        idcs = sub_df[['ants/x', 'ants/y', 'ants/z']].astype(int).values
        return [voxel_volume[tuple(i)] > 0 for i in idcs]

    # Load from tiff
    region_data = np.swapaxes(np.moveaxis(tifffile.imread(region_map), 0, 2), 0, 1)

    # Check coordinates
    df = rois.dataframe
    df[f'is_in_{area_name}'] = coordinates_in(df, region_data)

    df.commit()


if __name__ == '__main__':
    pass