"""
 @file : MixMat.py
 @author: G. Lehaut, LPC Caen, CNRS/IN2P3
 @date: 2023/10/15
 @description: this file contains different function tools to play with mixing matrix
  - defined_3flavor_PMNS_from_theta( theta_12, theta_23, theta_13, delta_cp = 0 )
    return a numpy array of 3x3 containing the PMNS matrix elements defined by the rotation matrix
  - defined_4flavor_PMNS_from_theta( theta_12, theta_23, theta_13, theta_14, theta_24, theta_34, delta_cp )
    return a numpy array of 4x4 containing the mixing matrix elements defined by the rotation matrix
  - defined_2flavor_MM_from_theta( theta )
    return a numpy array of 2x2 containing mixing matrix elements

"""

import numpy as np
from scipy import linalg


# ===================================================================
class ref_mass_splitting_3flavor:
    def __init__(self):
        self.delta_m21_NO = 7.39e-5
        self.delta_m32_NO = 2.449e-3
        self.delta_m21_NO_elow = 6.79e-5
        self.delta_m32_NO_elow = 2.358e-3
        self.delta_m21_NO_ehigh = 8.01e-5
        self.delta_m32_NO_ehigh = 2.544e-3
        self.delta_m21_IO = 7.39e-5
        self.delta_m32_IO = -2.509e-3
        self.delta_m21_IO_elow = 6.79e-5
        self.delta_m32_IO_elow = -2.603e-3
        self.delta_m21_IO_ehigh = 8.01e-5
        self.delta_m32_IO_ehigh = -2.416e-3

    def get_nominal_mass_NO(self):
        return np.array([0, self.delta_m21_NO, self.delta_m32_NO + self.delta_m21_NO])

    def get_nominal_mass_IO(self):
        return np.array([0, self.delta_m21_IO, self.delta_m32_NO + self.delta_m21_IO])

    def get_rnd_mass_NO(self, nstat=1):
        ran12 = np.random.rand(nstat) * (self.delta_m21_NO_ehigh - self.delta_m21_NO_elow) + self.delta_m21_NO_elow
        ran32 = np.random.rand(nstat) * (self.delta_m32_NO_ehigh - self.delta_m32_NO_elow) + self.delta_m32_NO_elow
        zero = np.zeros_like(ran12)
        return np.array([zero, ran12, ran32 + ran12])

    def get_rnd_mass_IO(self, nstat=1):
        ran12 = np.random.rand(nstat) * (self.delta_m21_IO_ehigh - self.delta_m21_IO_elow) + self.delta_m21_IO_elow
        ran32 = np.random.rand(nstat) * (self.delta_m32_IO_ehigh - self.delta_m32_IO_elow) + self.delta_m32_IO_elow
        zero = np.zeros_like(ran12)
        return np.array([zero, ran12, ran32 + ran12])


# ______________________________________________________________________________________________________________

class ref_PMNS_3flavor:
    def __init__(self):
        self.U_33_min = np.array([[0.76, 0.5, 0.13],
                                  [0.21, 0.42, 0.61],
                                  [0.18, 0.38, 0.40]])
        self.U_33_max = np.array([[0.85, 0.6, 0.16],
                                  [0.54, 0.7, 0.79],
                                  [0.58, 0.72, 0.78]])
        self.delta_cp_NO = 222.
        self.delta_cp_NO_elow = 141.
        self.delta_cp_NO_ehigh = 370.
        self.theta12_NO = 33.82
        self.theta12_NO_elow = 31.61
        self.theta12_NO_ehigh = 36.27
        self.theta13_NO = 8.61
        self.theta13_NO_elow = 8.22
        self.theta13_NO_ehigh = 8.99
        self.theta23_NO = 48.3
        self.theta23_NO_elow = 40.8
        self.theta23_NO_ehigh = 51.3
        self.delta_cp_IO = 285.
        self.delta_cp_IO_elow = 205.
        self.delta_cp_IO_ehigh = 364.
        self.theta12_IO = 33.82
        self.theta12_IO_elow = 31.61
        self.theta12_IO_ehigh = 36.27
        self.theta13_IO = 8.65
        self.theta13_IO_elow = 8.26
        self.theta13_IO_ehigh = 9.02
        self.theta23_IO = 48.6
        self.theta23_IO_elow = 41
        self.theta23_IO_ehigh = 51.5

    def get_nominal_angle_NO(self):
        return self.theta12_NO, self.theta23_NO, self.theta13_NO

    def get_nominal_angle_IO(self):
        return self.theta12_IO, self.theta23_IO, self.theta13_IO

    def get_rnd_angle_NO(self, nstat=1):
        ran12 = np.random.rand(nstat) * (self.theta12_NO_ehigh - self.theta12_NO_elow) + self.theta12_NO_elow
        ran13 = np.random.rand(nstat) * (self.theta13_NO_ehigh - self.theta13_NO_elow) + self.theta13_NO_elow
        ran23 = np.random.rand(nstat) * (self.theta23_NO_ehigh - self.theta23_NO_elow) + self.theta23_NO_elow
        randcp = np.random.rand(nstat) * (self.delta_cp_NO_ehigh - self.delta_cp_NO_elow) + self.delta_cp_NO_elow
        return np.array([ran12, ran23, ran13, randcp])

    def get_rnd_angle_IO(self, nstat=1):
        # TODO: add function with gaussian
        ran12 = np.random.rand(nstat) * (self.theta12_IO_ehigh - self.theta12_IO_elow) + self.theta12_IO_elow
        ran13 = np.random.rand(nstat) * (self.theta13_IO_ehigh - self.theta13_IO_elow) + self.theta13_IO_elow
        ran23 = np.random.rand(nstat) * (self.theta23_IO_ehigh - self.theta23_IO_elow) + self.theta23_IO_elow
        randcp = np.random.rand(nstat) * (self.delta_cp_IO_ehigh - self.delta_cp_IO_elow) + self.delta_cp_IO_elow
        return np.array([ran12, ran23, ran13, randcp])


def defined_3flavor_PMNS_from_theta(theta_12, theta_23, theta_13, delta_cp=0):
    ctheta_12 = np.cos(np.deg2rad(theta_12))
    stheta_12 = np.sin(np.deg2rad(theta_12))
    ctheta_23 = np.cos(np.deg2rad(theta_23))
    stheta_23 = np.sin(np.deg2rad(theta_23))
    ctheta_13 = np.cos(np.deg2rad(theta_13))
    stheta_13 = np.sin(np.deg2rad(theta_13))
    zero = np.zeros_like(theta_12)
    un = np.zeros_like(theta_12) + 1.
    if not isinstance(delta_cp, np.ndarray):
        dcp = un * np.exp(1j * np.deg2rad(delta_cp))
    else:
        dcp = np.exp(1j * delta_cp)
    Rcp = np.array([[un, zero, zero],
                    [zero, un, zero],
                    [zero, zero, dcp]])
    R12 = np.array([[ctheta_12, stheta_12, zero],
                    [-stheta_12, ctheta_12, zero],
                    [zero, zero, un]])
    # print(R12)
    R23 = np.array([[un, zero, zero],
                    [zero, ctheta_23, stheta_23],
                    [zero, -stheta_23, ctheta_23]])
    R13 = np.array([[ctheta_13, zero, stheta_13],
                    [zero, un, zero],
                    [-stheta_13, zero, ctheta_13]])

    if not isinstance(theta_12, np.ndarray):
        R = R23 @ Rcp @ R13 @ R12
    else:
        R12 = np.einsum('ijk->kij', R12)
        R13 = np.einsum('ijk->kij', R13)
        R23 = np.einsum('ijk->kij', R23)
        Rcp = np.einsum('ijk->kij', Rcp)
        R = R23 @ Rcp @ R13 @ R12
    return R


def defined_4flavor_from_theta(theta_12, theta_23, theta_13, theta_14, theta_24, theta_34, delta_cp=0):
    ctheta_12 = np.cos(np.deg2rad(theta_12))
    stheta_12 = np.sin(np.deg2rad(theta_12))
    ctheta_23 = np.cos(np.deg2rad(theta_23))
    stheta_23 = np.sin(np.deg2rad(theta_23))
    ctheta_13 = np.cos(np.deg2rad(theta_13))
    stheta_13 = np.sin(np.deg2rad(theta_13))

    ctheta_14 = np.cos(np.deg2rad(theta_14))
    stheta_14 = np.sin(np.deg2rad(theta_14))
    ctheta_24 = np.cos(np.deg2rad(theta_24))
    stheta_24 = np.sin(np.deg2rad(theta_24))
    ctheta_34 = np.cos(np.deg2rad(theta_34))
    stheta_34 = np.sin(np.deg2rad(theta_34))

    zero = np.zeros_like(theta_12)
    un = np.zeros_like(theta_12) + 1.
    if not isinstance(delta_cp, np.ndarray):
        dcp = un * np.exp(1j * np.deg2rad(delta_cp))
    else:
        dcp = np.exp(1j * delta_cp)
    Rcp = np.array([[un, zero, zero, zero],
                    [zero, un, zero, zero],
                    [zero, zero, dcp, zero],
                    [zero, zero, zero, un]])
    R12 = np.array([[ctheta_12, stheta_12, zero, zero],
                    [-stheta_12, ctheta_12, zero, zero],
                    [zero, zero, un, zero],
                    [zero, zero, zero, un]])
    # print(R12)
    R23 = np.array([[un, zero, zero, zero],
                    [zero, ctheta_23, stheta_23, zero],
                    [zero, -stheta_23, ctheta_23, zero],
                    [zero, zero, zero, un]])
    R13 = np.array([[ctheta_13, zero, stheta_13, zero],
                    [zero, un, zero, zero],
                    [-stheta_13, zero, ctheta_13, zero],
                    [zero, zero, zero, un]])
    R14 = np.array([[ctheta_14, zero, zero, stheta_14],
                    [zero, un, zero, zero],
                    [zero, zero, un, zero],
                    [-stheta_14, zero, zero, ctheta_14]])
    R24 = np.array([[un, zero, zero, zero],
                    [zero, ctheta_24, zero, stheta_24],
                    [zero, zero, un, zero],
                    [zero, -stheta_24, zero, ctheta_24]])
    R34 = np.array([[un, zero, zero, zero],
                    [zero, un, zero, zero],
                    [zero, zero, ctheta_34, stheta_34],
                    [zero, zero, -stheta_34, ctheta_34]])

    if not isinstance(theta_12, np.ndarray):
        R = R34 @ R24 @ R14 @ R23 @ Rcp @ R13 @ R12
    else:
        R12 = np.einsum('ijk->kij', R12)
        R13 = np.einsum('ijk->kij', R13)
        R23 = np.einsum('ijk->kij', R23)
        R14 = np.einsum('ijk->kij', R14)
        R24 = np.einsum('ijk->kij', R24)
        R34 = np.einsum('ijk->kij', R34)
        Rcp = np.einsum('ijk->kij', Rcp)
        R = R34 @ R24 @ R14 @ R23 @ Rcp @ R13 @ R12
    return R


def defined_2flavor_MM_from_theta(theta):
    ctheta = np.cos(np.deg2rad(theta))
    stheta = np.sin(np.deg2rad(theta))
    R = np.array([[ctheta, stheta],
                  [-stheta, ctheta]])
    return R


def check_U_prior_constaint(U, U_min, U_max):

    sel0 = (U[:, 0, 0:3] > U_min.reshape(1, 3, 3)[0, 0, 0:3])
    print(np.sum(sel0))
    if (np.sum(sel0) == 0):
        return
    sel1 = (U[:, 1, 0:3] > U_min.reshape(1, 3, 3)[0, 1, 0:3])
    print(np.sum(sel1))
    if (np.sum(sel1) == 0):
        return
    sel2 = (U[:, 2, 0:3] > U_min.reshape(1, 3, 3)[0, 2, 0:3])
    print(np.sum(sel2))


def check_U_prior_unitarity(U):
    print(U @ U.T)


def closest_unitary(A):
    """ Calculate the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix A.

        Return U as a numpy matrix.
    """
    V, __, Wh = linalg.svd(A)
    U = np.matrix(V.dot(Wh))
    return U
