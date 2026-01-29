import os
from typing import List, Tuple

import numpy as np


# Universal direction for a fish-centric spherical coordinate system
#   where (in deg) azimuth 0, elevation 0 is "front"
#                  azimuth 0, elevation 90 is "up"
#                  azimuth 90, elevation 0 is "right"
universal_directions = {'front': (0., 0.), 'up': (0., np.pi / 2), 'right': (np.pi / 2, 0.)}


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    elevation = np.arctan2(z, hxy)
    azimuth = -np.arctan2(y, x)

    return np.array([azimuth, elevation, r])


def sph2cart(azimuth, elevation, r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = -r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)

    return np.array([x, y, z])


def mollweide_projection(longitude, latitude, radius = 1.0, central_meridian = 0.0):
    x = radius * 2 * np.sqrt(2) / np.pi * (longitude - central_meridian) * np.cos(latitude)
    y = radius * np.sqrt(2) * np.sin(latitude)

    return x, y


def eckert_iv_projection(longitude, latitude, radius = 1.0, central_meridian = 0.0):
    x = 2 / np.sqrt(4 * np.pi + np.pi**2) * radius * (longitude - central_meridian) * (1 + np.cos(latitude))
    y = 2 * np.sqrt(np.pi / (4 + np.pi)) * radius * np.sin(latitude)

    return x, y


def rotmat_from_to(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Calculate rotation matrix for rotation between two 3d vectors
    based on https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    """
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)

    basis = np.cross(v2n, v1n)
    basis /= np.linalg.norm(basis)
    angle = np.arccos(np.dot(v1n, v2n))

    c = np.cos(angle)
    s = np.sin(angle)
    nc = 1 - c
    x, y, z = basis

    M = np.array([[c + x ** 2 * nc, x * y * nc - z * s, x * z * nc + y * s],
                  [y * x * nc + z * s, c + y ** 2 * nc, y * z * nc - x * s],
                  [z * x * nc - y * s, z * y * nc + x * s, c + z ** 2 * nc]])

    return M


def fisher_pdf(vec: np.ndarray, mu: np.ndarray, kappa: float):
    """Implementation for the von Mises-Fisher distribution in 3 dimensions
    (Fisher distribution) based on https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    """

    norm_c3 = lambda kappa: kappa / (2 * np.pi * (np.exp(kappa) - np.exp(-kappa)))
    pd = norm_c3(kappa) * np.exp(kappa * np.dot(vec, mu))

    return pd


def despine(axis, spines=None, hide_ticks=True):
    def hide_spine(spine):
        spine.set_visible(False)

    for spine in axis.spines.keys():
        if spines is not None:
            if spine in spines:
                hide_spine(axis.spines[spine])
        else:
            hide_spine(axis.spines[spine])

    if hide_ticks:
        axis.xaxis.set_ticks([])
        axis.yaxis.set_ticks([])


class PlatonicSolid:

    corners: List[List[float]]
    _faces: List[List[int]]

    def __init__(self, subdiv_lvl):

        # Calculate initial vertices
        self._vertices = [self._vertex(*v) for v in self.corners]
        self._vertex_lvls = [0] * len(self.corners)

        # Subdivide faces
        self._cache = None
        self.subdiv_lvl = subdiv_lvl
        self._subdivide()

    def get_indices(self):
        return np.ascontiguousarray(np.array(self._faces, dtype=np.uint32)).flatten()

    def get_vertices(self):
        return np.ascontiguousarray(np.array(self._vertices, dtype=np.float32))

    def get_vertex_levels(self):
        return np.ascontiguousarray(np.array(self._vertex_lvls, dtype=np.int32))

    def get_spherical_coordinates(self):
        _vertices = self.get_vertices()
        az, el = np.array(cart2sph(_vertices[0, :], _vertices[1, :], _vertices[2, :]))
        return np.ascontiguousarray(az), np.ascontiguousarray(el)

    def _vertex(self, x, y, z):
        vlen = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return [i/vlen for i in (x, y, z)]

    def _midpoint(self, p1, p2, v_lvl):
        key = '%i/%i' % (min(p1, p2), max(p1, p2))

        if key in self._cache:
            return self._cache[key]

        v1 = self._vertices[p1]
        v2 = self._vertices[p2]
        middle = [sum(i)/2 for i in zip(v1, v2)]

        self._vertices.append(self._vertex(*middle))
        self._vertex_lvls.append(v_lvl)
        index = len(self._vertices) - 1

        self._cache[key] = index

        return index

    def _subdivide(self):
        self._cache = {}
        for i in range(self.subdiv_lvl):
            new_faces = []
            for face in self._faces:
                v = [self._midpoint(face[0], face[1], i+1),
                     self._midpoint(face[1], face[2], i+1),
                     self._midpoint(face[2], face[0], i+1)]

                new_faces.append([face[0], v[0], v[2]])
                new_faces.append([face[1], v[1], v[0]])
                new_faces.append([face[2], v[2], v[1]])
                new_faces.append([v[0], v[1], v[2]])

            self._faces = new_faces


class Tetrahedron(PlatonicSolid):

    corners = [
        [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(3)],  # V0
        [-1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(3)],  # V1
        [0, -np.sqrt(2 / 3), 1 / np.sqrt(3)],  # V2
        [0, 0, -np.sqrt(3)]  # V3
    ]

    _faces = [
        [0, 1, 2],  # Face 1
        [0, 1, 3],  # Face 2
        [0, 2, 3],  # Face 3
        [1, 2, 3]  # Face 4
    ]


class Octahedron(PlatonicSolid):

    corners = np.array([
        [1, 0, 0],  # V0
        [-1, 0, 0],  # V1
        [0, 1, 0],  # V2
        [0, -1, 0],  # V3
        [0, 0, 1],  # V4
        [0, 0, -1]  # V5
    ])

    # Define the face indices of the octahedron
    _faces = np.array([
        [0, 2, 4],  # Face 0
        [0, 2, 5],  # Face 1
        [0, 3, 4],  # Face 2
        [0, 3, 5],  # Face 3
        [1, 2, 4],  # Face 4
        [1, 2, 5],  # Face 5
        [1, 3, 4],  # Face 6
        [1, 3, 5]  # Face 7
    ])


class IcosahedronSphere(PlatonicSolid):
    gr = (1 + np.sqrt(5)) / 2

    corners = [
        [-1, gr,  0],
        [1,  gr,  0],
        [-1, -gr, 0],
        [1,  -gr, 0],
        [0,  -1,  gr],
        [0,  1,   gr],
        [0,  -1,  -gr],
        [0,  1,   -gr],
        [gr, 0,   -1],
        [gr, 0,   1],
        [-gr, 0,  -1],
        [-gr, 0,  1],
    ]

    _faces = [

        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],

        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],

        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],

        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
