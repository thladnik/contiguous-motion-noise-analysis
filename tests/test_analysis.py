import numpy as np
import pytest

from cmn_analysis import entarchy_adapter, geometry, rf
from cmn_analysis.stimulus_data import CMNStimulus, from_session_data

from conftest import (PLANTED_AXIS, PLANTED_CENTER, SAMPLE_RATE, icosphere_faces, simulate_dff)


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def make_stimulus(shown, sphere, kind=None) -> CMNStimulus:
    """The recording of `shown` in the storage style of `kind` (default: its own)."""
    stimulus = CMNStimulus(sphere['centers'], sphere['vertices'])
    if (kind or shown['kind']) == 'original':
        stimulus.add_original(shown['frame_times'], shown['frame_index'], shown['field'], shown['end_time'])
    else:
        stimulus.add_realtime(shown['frame_times'], shown['field'][shown['frame_index']], shown['end_time'])
    return stimulus


# ------------------------------------------------------------------------------ stimulus data

def test_both_storage_styles_sample_identically(shown, sphere):
    times = np.arange(0., shown['end_time'] + 5., 1 / SAMPLE_RATE)
    sel_a, vec_a, ids_a = make_stimulus(shown, sphere, 'original').sample(times, offset=0.013)
    sel_b, vec_b, ids_b = make_stimulus(shown, sphere, 'realtime').sample(times, offset=0.013)
    assert np.array_equal(sel_a, sel_b) and np.array_equal(ids_a, ids_b)
    assert np.array_equal(vec_a, vec_b)
    assert not sel_a[0] and not sel_a[-1] and sel_a.sum() == pytest.approx(600 * SAMPLE_RATE, abs=2)


def test_sample_returns_the_frame_on_screen(sphere):
    centers = sphere['centers']
    vectors = np.arange(4, dtype=np.float32)[:, None, None] * np.ones((4, len(centers), 3), dtype=np.float32)
    stimulus = CMNStimulus(centers)
    stimulus.add_realtime([10., 11., 12., 13.], vectors, end_time=14.)
    selection, sampled, _ = stimulus.sample(np.array([9.9, 10., 10.99, 11., 13.5, 14.1]))
    assert selection.tolist() == [False, True, True, True, True, False]
    assert sampled[:, 0, 0].tolist() == [0., 0., 1., 3.]


def test_two_segments_keep_their_order(sphere):
    centers = sphere['centers']
    ones = np.ones((2, len(centers), 3), dtype=np.float32)
    stimulus = CMNStimulus(centers)
    stimulus.add_realtime([20., 21.], 2 * ones, end_time=22.)
    stimulus.add_original([0., 1.], [1, 0], np.stack([5 * ones[0], 7 * ones[0]]), end_time=2.)
    selection, sampled, ids = stimulus.sample(np.array([0.5, 1.5, 10., 20.5]))
    assert ids.tolist() == [1, 1, 0, 2]
    assert sampled[:, 0, 0].tolist() == [7., 5., 2.]


def test_session_data_loader_reads_both_stimuli(sphere):
    base = pytest.importorskip('vxpy.export.base')
    centers, vertices = sphere['centers'].astype(np.float32), sphere['vertices'].astype(np.float32)
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(50, len(centers), 3)).astype(np.float32)
    t_ns = (np.arange(200) * 1e9 / 60).astype(np.int64)
    index = np.arange(200) // 4

    def stream(name, data, t=None):
        t = np.zeros(1, dtype=np.int64) if t is None else t
        return base.StreamData(name, np.arange(len(t)), t, data)

    def session(kind):
        streams = {'disp.p_frame_index': stream('disp.p_frame_index', index[:, None], t_ns),
                   'disp.dump_CMN__centers': stream('disp.dump_CMN__centers', centers[None]),
                   'disp.dump_CMN__vertices': stream('disp.dump_CMN__vertices', vertices[None])}
        if kind == 'original':
            streams['disp.dump_CMN__motion_vectors'] = stream('disp.dump_CMN__motion_vectors', matrix[None])
        else:
            streams['disp.p_motion_vectors_icosahedron2'] = stream('disp.p_motion_vectors_icosahedron2',
                                                                   matrix[index], t_ns)
        data = base.SessionData(attrs={}, streams=streams)
        data.__dict__['runs'] = [{'node': 'disp', 'kind': 'visual', 'class': 'apps.x.visuals.cmn:CMN',
                                  'start_ns': int(t_ns[20]), 'end_ns': int(t_ns[180])},
                                 {'node': 'disp', 'kind': 'visual', 'class': 'vxpy.visuals:Pause',
                                  'start_ns': int(t_ns[181]), 'end_ns': int(t_ns[199])}]
        return data

    times = np.arange(0., 3.4, 0.05)
    results = {}
    for kind in ('original', 'realtime'):
        stimulus = from_session_data(session(kind))
        assert [s.kind for s in stimulus.segments] == [kind]
        results[kind] = stimulus.sample(times)
    assert np.array_equal(results['original'][0], results['realtime'][0])
    assert np.array_equal(results['original'][1], results['realtime'][1])
    assert results['original'][0].sum() > 40


# ---------------------------------------------------------------------------------- geometry

def test_face_adjacency_is_the_mesh(sphere):
    edges = geometry.face_adjacency(sphere['vertices'])
    assert len(edges) == 320 * 3 // 2
    assert np.all(np.bincount(edges.ravel()) == 3)
    assert np.array_equal(edges, geometry.nearest_adjacency(sphere['centers']))


def test_local_frames_are_orthonormal_even_at_the_pole():
    centers = np.concatenate([geometry.icosphere_axes(2), [[0., 0., 1.], [0., 0., -1.]]])
    for up in (geometry.UP, np.array([np.sin(0.3), 0., np.cos(0.3)])):
        horizontal, vertical = geometry.local_frames(centers, up)
        for a, b in ((horizontal, vertical), (horizontal, centers), (vertical, centers)):
            assert np.allclose(np.sum(a * b, axis=1), 0., atol=1e-9)
        assert np.allclose(np.linalg.norm(horizontal, axis=1), 1.) and np.allclose(np.linalg.norm(vertical, axis=1), 1.)


def test_solid_angles_cover_the_sphere(sphere):
    assert geometry.patch_solid_angles(sphere['vertices'], 320).sum() == pytest.approx(4 * np.pi)


# ------------------------------------------------------------------------------- statistics

def test_bin_significance_equals_ranking_every_sample():
    rng = np.random.default_rng(1)
    null = rng.random((200, 12, 8)).astype(np.float32)
    observed = rng.random((12, 8)).astype(np.float32)
    sig_observed, sig_null = rf.bin_significance(observed, null, alpha=0.05)

    def ranked(x):
        cdf = (x > null).sum(axis=0) / len(null)
        return (cdf > 0.975).astype(np.int8) - (cdf < 0.025).astype(np.int8)

    assert np.array_equal(sig_observed, ranked(observed))
    assert np.array_equal(sig_null, np.stack([ranked(n) for n in null]))


def test_clusters_do_not_depend_on_patch_order(sphere):
    rng = np.random.default_rng(2)
    edges = geometry.face_adjacency(sphere['vertices'])
    for _ in range(20):
        significant = rng.random((320, 16)) < 0.12
        scores, _, members = rf.find_clusters(significant, edges)
        order = rng.permutation(320)
        inverse = np.argsort(order)
        scores_p, _, members_p = rf.find_clusters(significant[order], inverse[edges])
        assert sorted(scores) == sorted(scores_p)
        assert np.array_equal(members[order], members_p)


def test_cluster_members_are_the_shared_bins():
    significant = np.zeros((4, 3), dtype=bool)
    significant[0, [0, 1]] = significant[1, [1, 2]] = significant[2, 2] = significant[3, 0] = True
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    scores, labels, members = rf.find_clusters(significant, edges)
    assert labels.tolist() == [0, 0, 0, -1]
    assert members.astype(int).tolist() == [[0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]]
    assert scores.tolist() == [4.]


def test_etas_match_the_one_hot_definition(sphere):
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(300, 320, 3)).astype(np.float32)
    binned = rf.BinnedStimulus.from_vectors(vectors, sphere['centers'], bin_num=16)
    weights = rng.random(300)

    local = geometry.to_local_2d(vectors, binned.horizontal, binned.vertical)
    angle, speed = np.arctan2(local[..., 1], local[..., 0]), np.linalg.norm(local, axis=-1)
    edges = np.linspace(-np.pi, np.pi, 17)
    one_hot = speed[..., None] * ((edges[:-1] <= angle[..., None]) & (angle[..., None] < edges[1:]))
    expected = np.einsum('t,tpk->pk', weights / weights.sum(), one_hot)
    assert np.allclose(binned.etas(weights), expected, atol=1e-5)


# --------------------------------------------------------------- end to end, both stimuli

@pytest.fixture(scope='module')
def mapped(shown, sphere):
    rng = np.random.default_rng(7)
    times, dff, planted = simulate_dff(shown, sphere['centers'], rng)
    stimulus = make_stimulus(shown, sphere)
    selection, vectors, _ = stimulus.sample(times)
    binned = rf.BinnedStimulus.from_vectors(vectors, stimulus.centers)
    edges = geometry.face_adjacency(stimulus.vertices)
    weights = rf.event_weights(dff, SAMPLE_RATE, selection)
    result = rf.map_rf(binned, weights[selection], edges, shift_num=300)
    return {'result': result, 'planted': planted, 'binned': binned, 'edges': edges,
            'selection': selection, 'weights': weights, 'rng': rng, 'times': times}


def test_planted_receptive_field_is_found(mapped, sphere):
    result, planted = mapped['result'], mapped['planted']
    assert result.has_receptive_field
    found = result.rf_patches
    assert (found & planted).sum() / planted.sum() > 0.6         # most of it is recovered
    center = sphere['centers'][found].mean(axis=0)
    assert angle_between(center, PLANTED_CENTER) < 20.

    # preferred directions inside the planted field point up
    upward = geometry.translation_templates(sphere['centers'], PLANTED_AXIS[None])[0]
    inside = found & planted
    cosines = np.sum(result.preferred_vectors_3d[inside] * upward[inside], axis=1) / (
        np.linalg.norm(result.preferred_vectors_3d[inside], axis=1) * np.linalg.norm(upward[inside], axis=1))
    assert np.median(cosines) > 0.9


def test_egomotion_fit_recovers_the_translation_axis(mapped, sphere):
    fit = rf.fit_egomotion(sphere['centers'], mapped['result'])
    assert fit['is_translation_selective']
    assert fit['translation_best_similarity'] > 0.8
    assert angle_between(fit['translation_axis'], PLANTED_AXIS) < 25.


def test_egomotion_fit_ignores_the_binning_frame(mapped, sphere, shown):
    """A pitch-corrected local frame changes the 2D numbers but not the fitted axis."""
    stimulus = make_stimulus(shown, sphere)
    _, vectors, _ = stimulus.sample(mapped['times'])
    pitched = rf.BinnedStimulus.from_vectors(vectors, stimulus.centers,
                                             up=np.array([np.sin(np.radians(15)), 0., np.cos(np.radians(15))]))
    result = rf.map_rf(pitched, mapped['weights'][mapped['selection']], mapped['edges'], shift_num=300)
    fit, reference = rf.fit_egomotion(sphere['centers'], result), rf.fit_egomotion(sphere['centers'], mapped['result'])
    assert angle_between(fit['translation_axis'], reference['translation_axis']) < 10.


def test_same_seed_same_result(mapped):
    again = rf.map_rf(mapped['binned'], mapped['weights'][mapped['selection']], mapped['edges'], shift_num=300)
    assert np.array_equal(again.rf_patches, mapped['result'].rf_patches)
    assert np.array_equal(again.cluster_p_values, mapped['result'].cluster_p_values)


def test_noise_rarely_has_a_receptive_field(mapped):
    """The test is calibrated at cluster_alpha = 5 % (measured: 3 of 60 noise neurons with 1000
    shifts), so a few of many noise neurons pass; most must not."""
    rng = np.random.default_rng(11)
    positives = 0
    for i in range(8):
        dff = np.convolve(rng.normal(size=len(mapped['times'])), np.ones(8) / 8, mode='same')
        weights = rf.event_weights(dff, SAMPLE_RATE, mapped['selection'])
        positives += rf.map_rf(mapped['binned'], weights[mapped['selection']], mapped['edges'],
                               shift_num=300, seed=i).has_receptive_field
    assert positives <= 2


def test_lag_profile_peaks_near_zero_lag(mapped):
    lags = np.arange(-30, 31, 5)
    profile = rf.lag_profile(mapped['binned'], mapped['weights'][mapped['selection']], lags, shift_num=100)
    assert abs(lags[np.argmax(profile)]) <= 10
    assert profile.max() > 1.15 * profile[[0, -1]].mean()   # RMS z over all 5120 bins: 1 means noise


# ---------------------------------------------------------------------- entarchy adapter

class Entity(dict):
    def __init__(self, data=None, **links):
        super().__init__(data or {})
        self.__dict__.update(links)


class Rois:
    def __init__(self):
        self.values = {}

    def __setitem__(self, key, value):
        self.values[key] = value


def fake_recording(shown, sphere, times):
    visual = 'CMN3D20240411'
    phase = Entity({'display/__visual_name': visual, '__time': shown['frame_times'][:, None],
                    'display/__start_time': shown['frame_times'][0],
                    'display/__target_duration': shown['end_time'] - shown['frame_times'][0]})
    rec = Entity({f'display/{visual}/centers_0': sphere['centers'][None],
                  f'display/{visual}/vertices_0': sphere['vertices'][None]})
    if shown['kind'] == 'original':
        rec[f'display/{visual}/motion_vectors_0'] = shown['field'][None]
        phase['frame_index'] = shown['frame_index'][:, None]
    else:
        phase['motion_vectors'] = shown['field'][shown['frame_index']]
    grating = Entity({'display/__visual_name': 'TranslationGrating'})

    rec.__dict__.update(id=1, phases=[grating, phase], layers=[Entity({'t_offset': 0.}, id=1)], rois=Rois(),
                        animal=Entity({'ants/init_x_rotation': 10.}), imaging=[0],
                        sole_imaging=lambda: {'frame_times': times})
    return rec


def test_adapter_runs_on_both_recording_styles(shown, sphere, mapped):
    rng = np.random.default_rng(7)
    times, dff, planted = simulate_dff(shown, sphere['centers'], rng)
    rec = fake_recording(shown, sphere, times)
    entarchy_adapter._binned_cache.clear()
    entarchy_adapter.process_recording(rec, sample_rate=SAMPLE_RATE)

    assert rec.rois.values == {'has_cmn_dataset': True}
    assert rec['cmn_stimulus_kinds'] == [shown['kind']]
    layer = rec.layers[0]
    assert layer['cmn_motion_vectors_3d'].shape == (layer['cmn_phase_selection'].sum(), 320, 3)
    assert layer['cmn_motion_vectors_3d'].dtype == np.float32

    resampled = np.interp(rec['time_resampled'], times, dff)
    roi = Entity({'dff_resampled': resampled}, recording=rec, layer=layer)
    entarchy_adapter.run_cmn_analysis(roi, bootstrap_num=300)

    assert roi['has_receptive_field']
    found = np.zeros(320, dtype=bool)
    found[roi['significant_patch_indices']] = True
    assert (found & planted).sum() / planted.sum() > 0.6
    assert roi['is_translation_selective']
    assert angle_between(roi['translation_axis'], PLANTED_AXIS) < 25.   # despite the 10 deg pitch frame
    assert roi['translation_vectors'].shape == roi['preferred_vectors'].shape == (320, 2)
    assert 0 < roi['rf_size_sr'] < 4 * np.pi
    assert len(roi['cluster_unique_patch_indices']) == len(roi['cluster_p_values'])
