"""Shared fixtures: a CMN field from the generator of each stimulus, a simulated neuron with a
planted receptive field, and the two ways a recording stores the stimulus.

The two generators below are the numerical cores of the vxPy visuals ``cmn_redesign.py``
(``create_motion_matrix``) and ``cmn_redesign_realtime.py`` (``MotionField``), copied here
without the rendering code so the tests need neither vxPy nor a GL context.
"""
import numpy as np
import pytest
from scipy import signal

from cmn_analysis import geometry

MOTION_FPS = 20
DISPLAY_FPS = 60
SAMPLE_RATE = 10.
DURATION = 600  # [s]
START = 5.      # CMN phase starts 5 s into the recording

# CR = 2 * sigma * sqrt(-ln 0.1)  (Zhang & Arrenberg 2019, Materials and methods)
CR_PER_SIGMA = 2 * np.sqrt(-np.log(0.1))
RAYLEIGH_SIGMA_PER_K = 0.1852


def icosphere_faces(subdivisions: int):
    """(vertices (3P, 3) with one copy per face, centers (P, 3))."""
    axes = geometry.icosphere_axes(0)
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (3, 9, 4), (3, 4, 2),
             (3, 2, 6), (3, 6, 8), (3, 8, 9), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
             (7, 1, 8), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    tri = axes[np.array(faces)]
    for _ in range(subdivisions):
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, bc, ca = (a + b), (b + c), (c + a)
        ab, bc, ca = (m / np.linalg.norm(m, axis=1, keepdims=True) for m in (ab, bc, ca))
        tri = np.concatenate([np.stack(t, axis=1) for t in
                              ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))])
    centers = tri.mean(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return tri.reshape((-1, 3)), centers


@pytest.fixture(scope='session')
def sphere():
    vertices, centers = icosphere_faces(2)
    return {'vertices': vertices, 'centers': centers}


# ------------------------------------------------------------------ the two CMN generators

def original_motion_matrix(centers: np.ndarray, frame_num: int, tp_cr: int = 20, sp_cr: float = 30.,
                           seed: int = 5) -> np.ndarray:
    """``create_motion_matrix`` of the original visual: white noise unit vectors, smoothed in
    time and over great-circle distance with Gaussian kernels of the given contiguity radii."""
    tp_sigma, sp_sigma = tp_cr / CR_PER_SIGMA, sp_cr / CR_PER_SIGMA
    flow = np.random.RandomState(seed).normal(size=(len(centers), frame_num, 3))  # legacy stream, as the visual
    flow /= np.linalg.norm(flow, axis=-1)[:, :, None]

    taps = np.linspace(-2 * tp_cr, 2 * tp_cr, num=4 * tp_cr + 1)
    tp_kernel = np.exp(-0.5 * (taps / tp_sigma) ** 2) / (tp_sigma * np.sqrt(2 * np.pi))
    smooth = np.stack([signal.convolve(flow[:, :, i], tp_kernel[None, :], mode='same') for i in range(3)], axis=-1)

    chord = np.minimum(np.linalg.norm(centers[:, None] - centers[None], axis=-1), 2.)
    angle = np.degrees(2 * np.arcsin(chord / 2))
    sp_kernel = np.exp(-0.5 * (angle / sp_sigma) ** 2) / (sp_sigma * np.sqrt(2 * np.pi))
    sp_kernel *= sp_kernel > .001
    return np.einsum('pq,qti->tpi', sp_kernel, smooth)


class MotionField:
    """``MotionField`` of the realtime visual: every frame is a pure function of (seed, index)."""

    def __init__(self, centers, sp_cr, tp_cr_frames, max_velocity, seed=1):
        self.centers, self.seed = centers, int(seed)
        taps = np.arange(-2 * tp_cr_frames, 2 * tp_cr_frames + 1)
        self.tp_kernel = np.exp(-0.5 * (taps / (tp_cr_frames / CR_PER_SIGMA)) ** 2)
        self.half_width = (len(self.tp_kernel) - 1) // 2
        chord = np.minimum(np.linalg.norm(centers[:, None] - centers[None], axis=-1), 2.)
        angle = np.degrees(2 * np.arcsin(chord / 2))
        kernel = np.exp(-0.5 * (angle / (sp_cr / CR_PER_SIGMA)) ** 2)
        kernel *= kernel > .001
        self.sp_kernel = kernel / np.linalg.norm(kernel, axis=1, keepdims=True)
        self.velocity_scale = np.radians(RAYLEIGH_SIGMA_PER_K * max_velocity) / np.linalg.norm(self.tp_kernel)

    def noise_frame(self, index: int) -> np.ndarray:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        return rng.normal(size=(len(self.centers), 3))

    def frames(self, frame_num: int) -> np.ndarray:
        noise = np.stack([self.noise_frame(i) for i in range(frame_num + 2 * self.half_width)])
        out = np.empty((frame_num, len(self.centers), 3))
        for t in range(frame_num):
            window = noise[t:t + 2 * self.half_width + 1]
            out[t] = self.sp_kernel @ np.tensordot(self.tp_kernel, window, axes=1)
        return out * self.velocity_scale


def _field(kind: str, centers: np.ndarray) -> np.ndarray:
    """(frames, P, 3) motion vectors [rad/s] from the generator of the given stimulus."""
    frame_num = DURATION * MOTION_FPS
    if kind == 'realtime':
        motion = MotionField(centers, sp_cr=30., tp_cr_frames=MOTION_FPS, max_velocity=67., seed=5)
        return motion.frames(frame_num).astype(np.float32)
    matrix = original_motion_matrix(centers, frame_num, tp_cr=MOTION_FPS, sp_cr=30.)
    return (matrix * np.radians(67.) / np.linalg.norm(matrix, axis=-1).max()).astype(np.float32)


@pytest.fixture(scope='session', params=['original', 'realtime'])
def shown(request, sphere):
    """What the display did: frame times, the index of the motion frame on each display frame,
    and the field, from the stimulus named by ``kind``."""
    field = _field(request.param, sphere['centers'])
    frame_times = START + np.arange(DURATION * DISPLAY_FPS) / DISPLAY_FPS
    frame_index = ((frame_times - START) * MOTION_FPS + 1e-9).astype(np.int64)
    return {'kind': request.param, 'field': field, 'frame_times': frame_times,
            'frame_index': frame_index, 'end_time': START + DURATION}


# ------------------------------------------------------------------- the simulated neuron

PLANTED_CENTER = np.array([np.cos(np.radians(20)) * np.cos(np.radians(-50)),
                           np.cos(np.radians(20)) * np.sin(np.radians(-50)),
                           np.sin(np.radians(20))])
PLANTED_AXIS = np.array([0., 0., 1.])  # receptive field prefers upward translation flow
PLANTED_RADIUS = 40.  # [deg]


def simulate_dff(shown, centers, rng, noise=0.15):
    """dF/F at the imaging rate of a neuron that sums upward motion within 40 deg of
    PLANTED_CENTER, through a GCaMP-like kernel. Returns (times, dff, planted patch mask)."""
    planted = np.degrees(np.arccos(np.clip(centers @ PLANTED_CENTER, -1, 1))) < PLANTED_RADIUS
    preferred = geometry.translation_templates(centers, PLANTED_AXIS[None])[0]
    preferred /= np.linalg.norm(preferred, axis=1, keepdims=True)

    times = np.arange(0., START + DURATION + 5., 1 / SAMPLE_RATE)
    frames = np.clip(((times - START) * MOTION_FPS).astype(int), 0, len(shown['field']) - 1)
    drive = np.einsum('tpi,pi->t', shown['field'][frames][:, planted], preferred[planted])
    drive[(times < START) | (times > START + DURATION)] = 0.
    rate = np.clip(drive - np.percentile(drive, 70), 0., None)

    t = np.arange(0., 8., 1 / SAMPLE_RATE)
    kernel = np.exp(-t / 1.5) - np.exp(-t / 0.2)
    dff = np.convolve(rate, kernel)[:len(times)]
    dff = dff / dff.std() + noise * rng.normal(size=len(times))
    return times, dff, planted
