import numpy as np


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


class neutrino:
    def __init__(self, mass_values, mixing_matrix, sign=1.):
        self.U = np.matrix(mixing_matrix)
        self.m_matrix = np.diag(mass_values)
        self.pure_state = np.identity(len(mass_values))
        self.sign = sign
        if sign != 1.:
            self.U = np.conjugate(self.U.T)

    def propagate(self, nu, E, length):
        vec = np.arange(len(nu))
        # factor 5.07617=(1km/hbar*c/GeV)) =>hbar c = 197 MeV.fm
        M = np.diag(np.exp(-1j * 5.07614 / (2. * E) * np.diag(self.m_matrix) * length))
        H = self.U @ M @ np.transpose(self.U.conjugate())
        p = []
        nul = H @ nu
        for j in vec:
            # print(l, neut.pure_state[j,:])
            p.append(np.absolute(nul[0, j]) ** 2)
        return p


# flavor 0, 1, 2
def propagate_vacuum(E_range, theta_range, theta12, theta13, theta23, m21, m31, delta_cp, flavor):
    # define masses
    U_PMNS = defined_3flavor_PMNS_from_theta(theta12, theta23, theta13, delta_cp)
    mass_square = np.array([0., m21, m31])
    neut = neutrino(mass_square, U_PMNS)
    anti_neut = neutrino(mass_square, U_PMNS, -1)

    E_range_x, theta_range_y = np.meshgrid(E_range, theta_range)
    E_range_x = E_range_x.flatten()
    theta_range_y = theta_range_y.flatten()

    p = []
    p_anti = []
    for i, e in enumerate(E_range_x):
        p.append(
            neut.propagate(neut.pure_state[flavor, :], E_range_x[i], 6386. * 2. * np.cos(np.deg2rad(theta_range_y[i]))))
        p_anti.append(anti_neut.propagate(neut.pure_state[flavor, :], E_range_x[i],
                                          6386. * 2. * np.cos(np.deg2rad(theta_range_y[i]))))
    return p, p_anti


def get_oscillation_maps_vacuum(osc_pars):
    """Get the 9 oscillation maps as a 3x3 np.array given:
     :param osc_pars: Oscillation parameters in order: [theta12, theta23, theta13, delta_cp, m21, m31]"""
    # Define grid
    E_range_in = np.logspace(0., 3., 120)
    theta_range_in = np.linspace(0., 90., 120)

    maps = []
    for fl in range(3):
        p_map, _ = propagate_vacuum(E_range=E_range_in,
                                    theta_range=theta_range_in,
                                    theta12=osc_pars[0],
                                    theta23=osc_pars[1],
                                    theta13=osc_pars[2],
                                    delta_cp=osc_pars[3],
                                    m21=osc_pars[4],
                                    m31=osc_pars[5],
                                    flavor=fl)
        maps.append(np.reshape(np.array(p_map), (int(len(p_map)**.5), int(len(p_map)**.5), 3, 1)))

    return np.concatenate(maps, -1)


if __name__ == '__main__':
    # Define parameters
    osc_pars_in = [2.392e+02,  2.955e+02,  2.183e+02,  2.128e+02,  6.523e-05, -5.502e-03]
    # Propagate
    maps_out = get_oscillation_maps_vacuum(osc_pars=osc_pars_in)

    from OscillationMaps.utils import plot_osc_maps
    plot_osc_maps(maps_out, title='Oscillation Maps in Vacuum')
