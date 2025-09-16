import sys
import numpy as np
from scipy.optimize import fsolve
from TIRO import Model

def equations(u,psi,zeta,nu):    
    (r,epsilon) = u
    pot = potential(psi,zeta,nu,r,epsilon)
    grad = gradient(psi,zeta,nu,r,epsilon)
    return pot, grad

def potential(psi,zeta,nu,r,epsilon):
    model = Model([psi,epsilon,zeta,nu])
    model.integrate()
    pot = model.external_solution(r,np.pi/2,0)
    return pot

def gradient(psi,zeta,nu,r,epsilon):
    model = Model([psi,epsilon,zeta,nu])
    model.integrate()
    grad = model.external_gradient(r,np.pi/2,0)
    return grad

if __name__ == "__main__":

    args = sys.argv
    if (len(args) != 4):
        print("Usage python Critical.py concentration asynchronicity galaxy_coefficient")
        sys.exit(1)
    psi = float(args[1]) #concentration
    zeta = float(args[2]) #asynchronicity parameter
    nu = float(args[3]) #galactic potential coefficient

    epsilon_estimate = 0.0005
    r_tidal_estimate = 15
    r_t, epsilon_crit = fsolve(equations,(r_tidal_estimate,epsilon_estimate),args=(psi,zeta,nu))
   
    print("The critical tidal strength parameter is " + str(np.floor(epsilon_crit*10**8)/10**8))