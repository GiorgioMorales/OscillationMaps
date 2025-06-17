from OscillationMaps.GenerateMaps.MixMat import *
from OscillationMaps.GenerateMaps.Neutrino import *


def propagate_vacuum(theta_12, theta_23, theta_13, delta_cp, m_21, m_31):
    """
    Produce the 9 oscillation maps in vacuum given:
    :param theta_12:
    :param theta_23:
    :param theta_13:
    :param delta_cp:
    :param m_21:
    :param m_31:
    :return: Osc. maps in vacuum np.array (3x3)
    """

    # Define grid
    E_range = np.logspace(0., 2., 120)
    theta_range = np.linspace(0., 90., 120)
    E_range_x, theta_range_y = np.meshgrid(E_range, theta_range)
    E_range_x = E_range_x.flatten()
    theta_range_y = theta_range_y.flatten()

    # Load PMNS 3 flavor priors
    ref_3flavor = ref_PMNS_3flavor()

    U_nominal_PMNS = defined_3flavor_PMNS_from_theta(theta_12=theta_12, theta_23=theta_23, theta_13=theta_13,
                                                     delta_cp=delta_cp)
    mass_square = [0., m_21, m_31]
    neut_3_nominal = neutrino(mass_square, U_nominal_PMNS)

    for i, l in enumerate(L):
        p = neut_3_nominal.propagate(nu_0,E,l)
        p_e_NO[i] = p[0]
        p_mu_NO[i] = p[1]
        p_tau_NO[i] = p[2]