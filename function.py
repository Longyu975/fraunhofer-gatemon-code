"""
Copyright (c) 2018 and later, Muhammad Irfan and Anton Akhmerov.
Copyright (c) 2024 Longyu Ma.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
"""



from functools import partial
import kwant
from kwant.digest import uniform
import numpy as np
import scipy.linalg as la
from scipy import integrate

def make_system_wire(a, W, L, pot):

    def onsite(site, par):
        (x, y) = site.pos
        Vg = par.Vg  #uniform
        dV = par.dV
        disorder = 0.5 * par.U0 * ((2 * uniform(repr(site), salt=par.seed)) - 1)
        
        return 4 * par.t - par.mu - disorder-pot(site,Vg, dV,a,W)

    # def hopx(site1, site2, par):
    #     xt, yt = site1.pos
    #     xs, ys = site2.pos
    #     phase = np.exp(-0.5 * np.pi * 1j * par.flux * (xt - xs) * (yt + ys))
    #     return -par.t * phase

    def hopx(site1, site2, par):
        xt, yt = site1.pos
        xs, ys = site2.pos
        yc = par.yc
        # Define the center in the y-direction
        phase = np.exp(-0.5 * np.pi * 1j * par.flux * (xt -xs) * (yt + ys-yc*2))
        return -par.t * phase

    def hopy(site1, site2, par):
        return -par.t

    lat = kwant.lattice.square(a, norbs=1)
    syst = kwant.Builder()
    syst[(lat(x, y) for x in range(L) for y in range(W))] = onsite
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy

    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)), time_reversal=1)

    def lead_hopx(site1, site2, par):
        return -par.t

    def lead_onsite(site, par):
        Vg = par.Vg
        return 4 * par.t - par.mu-Vg

    lead[(lat(0, j) for j in range(W))] = lead_onsite
    lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = lead_hopx
    lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    syst = syst.finalized()
    return syst

def energy_level_tight_binding(smatrix, phi, Delta):

    N, M = [len(li.momenta) // 2 for li in smatrix.lead_info]
    s = smatrix.data
    r_a11 = 1j * np.eye(N)
    r_a12 = np.zeros((N, M))
    r_a21 = r_a12.T
    r_a22 = 1j * np.exp(- 1j * phi) * np.eye(M)
    r_a = np.bmat([[r_a11, r_a12], [r_a21, r_a22]])

    # Matrix
    A = (r_a.dot(s) + (s.T).dot(r_a)) / 2

    eigVl, eigVc = la.eigh(A.T.conj().dot(A))

    energy_positive = Delta * np.sqrt(np.maximum(eigVl, 1e-12))
    energy_negative = -Delta * np.sqrt(np.maximum(eigVl, 1e-12))
    return energy_positive , energy_negative

# def supercurrent_tight_binding(smatrix, phi, Delta):
#     """Returns the supercurrent in a SNS Josephson junction using
#     a tight-binding model.

#     Parameters
#     ----------
#     smatrix : kwant.smatrix object
#         Contains scattering matrix and information of lead modes.
#     phi : float
#         Superconducting phase difference between two superconducting leads.
#     Delta : float
#         Superconducting gap.
#     """
#     N, M = [len(li.momenta) // 2 for li in smatrix.lead_info]
#     s = smatrix.data
#     r_a11 = 1j * np.eye(N)
#     r_a12 = np.zeros((N, M))
#     r_a21 = r_a12.T
#     r_a22 = 1j * np.exp(- 1j * phi) * np.eye(M)
#     r_a = np.bmat([[r_a11, r_a12], [r_a21, r_a22]])
#     # Matrix
#     A = (r_a.dot(s) + (s.T).dot(r_a)) / 2
#     # dr_a/dphi
#     dr_a11 = np.zeros((N, N))
#     dr_a12 = np.zeros((N, M))
#     dr_a21 = dr_a12.T
#     dr_a22 = np.exp(-1j * phi) * np.eye(M)
#     dr_a = np.bmat([[dr_a11, dr_a12], [dr_a21, dr_a22]])
#     # dA/dphi
#     dA = (dr_a.dot(s) + (s.T).dot(dr_a)) / 2
#     # d(A^dagger*A)/dphi
#     Derivative = (dA.T.conj()).dot(A) + (A.T.conj()).dot(dA)
#     Derivative = np.array(Derivative)
#     eigVl, eigVc = la.eigh(A.T.conj().dot(A))
#     eigVl = Delta * eigVl ** 0.5
#     eigVc = eigVc.T
#     current = np.sum((eigVc.T.conj().dot(Derivative.dot(eigVc)) / eigVl)
#                      for eigVl, eigVc in zip(eigVl, eigVc))
#     current = 2 * Delta * current.real
#     return current

def supercurrent_tight_binding(smatrix, phi, Delta):
    """Returns the supercurrent in a SNS Josephson junction using
    a tight-binding model.

    Parameters
    ----------
    smatrix : kwant.smatrix object
        Contains scattering matrix and information of lead modes.
    phi : float
        Superconducting phase difference between two superconducting leads.
    Delta : float
        Superconducting gap.
    """
    N, M = [len(li.momenta) // 2 for li in smatrix.lead_info]
    s = smatrix.data
    r_a11 = 1j * np.eye(N)
    r_a12 = np.zeros((N, M))
    r_a21 = r_a12.T
    r_a22 = 1j * np.exp(- 1j * phi) * np.eye(M)
    r_a = np.bmat([[r_a11, r_a12], [r_a21, r_a22]])
    # Matrix
    A = (r_a.dot(s) + (s.T).dot(r_a)) / 2
    # dr_a/dphi
    dr_a11 = np.zeros((N, N))
    dr_a12 = np.zeros((N, M))
    dr_a21 = dr_a12.T
    dr_a22 = np.exp(-1j * phi) * np.eye(M)
    dr_a = np.bmat([[dr_a11, dr_a12], [dr_a21, dr_a22]])
    # dA/dphi
    dA = (dr_a.dot(s) + (s.T).dot(dr_a)) / 2
    # d(A^dagger*A)/dphi
    Derivative = (dA.T.conj()).dot(A) + (A.T.conj()).dot(dA)
    Derivative = np.array(Derivative)
    eigVl, eigVc = la.eigh(A.T.conj().dot(A))
    eigVl = Delta * (eigVl ** 0.5)
    eigVc = eigVc.T
    current = np.sum((eigVc.T.conj().dot(Derivative.dot(eigVc)) / eigVl)
                     for eigVl, eigVc in zip(eigVl, eigVc))
    current = (1.6*1e-19)*(Delta)*210 * current.real/(0.66*1e-15)
    return current

def andreev_states(smatrix, phi, Delta):
    """Returns Andreev eigenvalues and eigenvectors.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem object
        A finalized kwant system having a scattering region
        connected with two semi-infinite leads.
    par : SimpleNamespace object
        Simplenamespace object with Hamiltonian parameters.
    phi : float
        Superconducting phase difference between the two superconducting leads.
    Delta : float
        Superconducting gap.
    """
    s = smatrix
    N, M = [len(li.momenta) // 2 for li in s.lead_info]
    s = s.data
    r_a11 = 1j * np.eye(N)
    r_a12 = np.zeros((N, M))
    r_a21 = r_a12.T
    r_a22 = 1j * np.exp(- 1j * phi) * np.eye(M)
    r_a = np.bmat([[r_a11, r_a12], [r_a21, r_a22]])
    zeros = np.zeros(shape=(len(s), len(s)))
    matrix = np.bmat([[zeros, (s.T.conj()).dot(r_a.conj())],
                      [(s.T).dot(r_a), zeros]])
    eigVl, eigVc = la.eig(matrix)
    eigVc = la.qr(eigVc)[0]
    eigVl = eigVl * Delta
    values = []
    vectors = []
    for ii in range(len(eigVl)):
        if eigVl[ii].real > 0 and eigVl[ii].imag > 0:
            values.append(eigVl[ii].real)
            vectors.append(eigVc.T[ii][0:len(eigVl) // 2])
    values = np.array(values)
    vectors = np.array(vectors)
    return values, vectors

def calculate_V_phi(smatrix, phi_value, Delta):
    """
    Calculate the potential V(phi) for a single phi_value.

    Parameters:
        smatrix: The scattering matrix obtained from kwant.smatrix.
        phi_value: A single value of phi for which to calculate V(phi).
        Delta: Parameter for scaling energy levels.

    Returns:
        V_phi: The calculated potential V(phi) for the given phi_value.
    """
    # Get the energy levels for the given phi_value
    energy_pos, energy_neg = energy_level_tight_binding(smatrix=smatrix, phi=phi_value, Delta=Delta)

    # Calculate V(phi)
    V_phi = np.sum(energy_neg)

    return V_phi