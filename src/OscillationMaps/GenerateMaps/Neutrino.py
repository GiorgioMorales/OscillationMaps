"""
 @file : Neutrino.py
 @author: G. Lehaut, LPC Caen, CNRS/IN2P3
 @date: 2023/10/16
 @description:

"""

import numpy as np


class neutrino:
    def __init__(self, mass_values, mixing_matrix, sign=1.):
        self.U = np.matrix(mixing_matrix)
        self.m_matrix = np.diag(mass_values)
        self.pure_state = np.identity(len(mass_values))
        self.sign = sign
        if sign != 1.:
            self.U = np.conjugate(self.U)
            
    def propagate(self, nu, E, length):
        vec = np.arange(len(nu))
        M = np.diag(np.exp(-1j*5.07614/(2.*E)*np.diag(self.m_matrix)*length))
        H = self.U @ M @ np.transpose(self.U.conjugate()) 
        p = []
        nul = H @ nu
        for j in vec:
            p.append(np.absolute(nul[0, j])**2)
        return p
