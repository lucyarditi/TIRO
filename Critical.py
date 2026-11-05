import argparse
import numpy as np
import sigfig
from scipy.optimize import fsolve

from TIRO import Model

def equations(u, psi, zeta, nu):    
    (r, epsilon) = u
    pot = potential(psi, zeta, nu, r,epsilon)
    grad = gradient(psi, zeta, nu, r, epsilon)
    return pot, grad

def potential(psi, zeta, nu, r, epsilon):
    model = Model([psi, epsilon, zeta, nu])
    model.integrate()
    return model.external_potential(r, np.pi/2, 0)

def gradient(psi, zeta, nu, r, epsilon):
    model = Model([psi ,epsilon, zeta, nu])
    model.integrate()
    return model.external_dr(r, np.pi/2, 0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("psi", help="concentration", type=float)
    parser.add_argument("zeta", help="asynchronicity aprameter", type=float)
    parser.add_argument("nu", help="galactic potential coefficient", type=float)
    args = parser.parse_args()

    epsilon_estimate = 0.0005
    r_tidal_estimate = 15
    r_t, epsilon_crit = fsolve(equations, (r_tidal_estimate,epsilon_estimate), args=(args.psi,args.zeta,args.nu))
   
    sf = 6                                                               #Round to a minimum of 6 significant figures.
    while sigfig.round(epsilon_crit,sigfigs=sf) > epsilon_crit: sf += 1  #Increase number of significant figures if rounding exceeds criticality.
    print(f"The critical tidal strength parameter (to first order) is {"{:e}".format(sigfig.round(epsilon_crit,sigfigs=sf))}")