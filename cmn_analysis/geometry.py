"""Sphere geometry for the CMN analysis: local tangent frames, patch adjacency, patch areas
and egomotion flow templates. Plain numpy, no framework imports."""
from typing import Optional, Tuple

import numpy as np

UP = np.array([0., 0., 1.])


def local_frames(centers: np.ndarray, up: np.ndarray = UP) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal tangent basis ``(horizontal, vertical)`` at each patch center, each ``(P, 3)``.

    ``vertical`` is the tangential part of `up`, ``horizontal = -(vertical x center)``. A patch
    whose center is parallel to `up` has no defined vertical; it gets an arbitrary but valid
    basis instead of NaNs (its direction bins are then arbitrary but self-consistent, because
    the same basis maps preferred directions back to 3D).
    """
    centers = np.asarray(centers, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)

    vertical = up - centers * (centers @ up)[:, None]
    norm = np.linalg.norm(vertical, axis=1)
    degenerate = norm < 1e-6
    if degenerate.any():
        # Any direction not parallel to the center does
        helper = np.where(np.abs(centers[degenerate, :1]) < 0.9, [[1., 0., 0.]], [[0., 1., 0.]])
        fallback = helper - centers[degenerate] * np.sum(helper * centers[degenerate], axis=1)[:, None]
        vertical[degenerate] = fallback
        norm[degenerate] = np.linalg.norm(fallback, axis=1)
    vertical /= norm[:, None]

    horizontal = -np.cross(vertical, centers)
    horizontal /= np.linalg.norm(horizontal, axis=1)[:, None]
    return horizontal, vertical


def to_local_2d(vectors: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    """Project 3D vectors ``(..., P, 3)`` into the local frames: ``(..., P, 2)``."""
    return np.stack([np.sum(vectors * horizontal, axis=-1),
                     np.sum(vectors * vertical, axis=-1)], axis=-1)


def to_global_3d(vectors_2d: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_local_2d` for tangential vectors: ``(..., P, 2)`` -> ``(..., P, 3)``."""
    return vectors_2d[..., 0:1] * horizontal + vectors_2d[..., 1:2] * vertical


def face_adjacency(vertices: np.ndarray) -> np.ndarray:
    """Pairs ``(E, 2)`` of faces that share an edge.

    `vertices` holds one vertex copy per face, ``(3 P, 3)`` or ``(P, 3, 3)``, as the CMN
    visuals dump it. Shared corners are matched by position.
    """
    tri = np.asarray(vertices, dtype=np.float64).reshape((-1, 3, 3))
    _, vertex_ids = np.unique(np.round(tri.reshape((-1, 3)), 5), axis=0, return_inverse=True)
    vertex_ids = vertex_ids.reshape((-1, 3))

    face = np.repeat(np.arange(len(tri)), 3)
    edge = np.sort(np.stack([vertex_ids[:, [0, 1, 2]].ravel(), vertex_ids[:, [1, 2, 0]].ravel()], axis=1), axis=1)
    order = np.lexsort((edge[:, 1], edge[:, 0]))
    edge, face = edge[order], face[order]
    shared = np.all(edge[1:] == edge[:-1], axis=1)
    pairs = np.stack([face[:-1][shared], face[1:][shared]], axis=1)
    return np.unique(np.sort(pairs, axis=1), axis=0)


def nearest_adjacency(centers: np.ndarray, neighbour_num: int = 3) -> np.ndarray:
    """Fallback when no mesh is available: each patch linked to its nearest neighbours,
    symmetrised. Matches edge adjacency on triangle-face geometries; on other layouts it is
    only an approximation."""
    distance = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    nearest = np.argsort(distance, axis=1)[:, 1:neighbour_num + 1]
    pairs = np.stack([np.repeat(np.arange(len(centers)), neighbour_num), nearest.ravel()], axis=1)
    return np.unique(np.sort(pairs, axis=1), axis=0)


def patch_solid_angles(vertices: Optional[np.ndarray], patch_num: int) -> np.ndarray:
    """Solid angle [sr] of each triangular patch; equal shares of the sphere if no mesh is known."""
    if vertices is None:
        return np.full(patch_num, 4 * np.pi / patch_num)
    tri = np.asarray(vertices, dtype=np.float64).reshape((-1, 3, 3))
    tri = tri / np.linalg.norm(tri, axis=2, keepdims=True)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    numerator = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c)))
    denominator = (1 + np.einsum('ij,ij->i', a, b) + np.einsum('ij,ij->i', b, c)
                   + np.einsum('ij,ij->i', a, c))
    return 2 * np.arctan2(numerator, denominator)


def icosphere_axes(subdivisions: int = 3) -> np.ndarray:
    """Roughly uniform candidate axes on the sphere: vertices of a subdivided icosahedron
    (642 for three subdivisions)."""
    g = (1 + np.sqrt(5)) / 2
    vertices = [(-1, g, 0), (1, g, 0), (-1, -g, 0), (1, -g, 0), (0, -1, g), (0, 1, g),
                (0, -1, -g), (0, 1, -g), (g, 0, -1), (g, 0, 1), (-g, 0, -1), (-g, 0, 1)]
    vertices = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in vertices]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (3, 9, 4), (3, 4, 2),
             (3, 2, 6), (3, 6, 8), (3, 8, 9), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
             (7, 1, 8), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdivisions):
        midpoints, new_faces = {}, []

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in midpoints:
                m = vertices[i] + vertices[j]
                vertices.append(m / np.linalg.norm(m))
                midpoints[key] = len(vertices) - 1
            return midpoints[key]

        for f0, f1, f2 in faces:
            m01, m12, m20 = midpoint(f0, f1), midpoint(f1, f2), midpoint(f2, f0)
            new_faces += [(f0, m01, m20), (f1, m12, m01), (f2, m20, m12), (m01, m12, m20)]
        faces = new_faces
    return np.array(vertices)


def translation_templates(centers: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Flow ``(A, P, 3)`` for the visual scene translating along each axis: the tangential
    part of the axis at every patch. Magnitude is sin(angle to the axis), zero at the poles of
    the flow field."""
    return axes[:, None, :] - centers[None] * (axes @ centers.T)[:, :, None]


def rotation_templates(centers: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Flow ``(A, P, 3)`` for rotation about each axis, ``-(axis x center)`` (the sign used by
    the existing pipeline). Magnitude is sin(angle to the axis)."""
    return -np.cross(axes[:, None, :], centers[None])


def field_similarity(templates: np.ndarray, field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cosine similarity ``(A,)`` between each template and `field` ``(P, 3)`` over the masked
    patches, taken over the whole vector field at once.

    Unlike a mean of per-patch cosines this weights each patch by both magnitudes, so patches
    near a template's singularity (where its direction is ill defined) and patches with a weak
    preferred vector count little, and no patch can produce a NaN.
    """
    t = templates[:, mask].reshape((templates.shape[0], -1))
    f = field[mask].ravel()
    denominator = np.linalg.norm(t, axis=1) * np.linalg.norm(f)
    return np.divide(t @ f, denominator, out=np.zeros(len(t)), where=denominator > 0)
