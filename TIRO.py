import sys
import math as m
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.special import gammainc, gamma

class Model(object):

    def __init__(self,concentration,epsilon,zeta,nu):
        self.param = np.array([concentration,epsilon,zeta,nu])
        offset = 10**-6 #dimensionless radius at which integration will start
        self.r = np.linspace(offset,100,10**7)  #dimensionless radii at which solution will be recorded

        """ central boundary conditions, corrected using Frobenius """
        self.bcs = np.zeros(6)
        self.bcs[0] = self.param[0]-((3/2)*(offset**2)) #ψ₀(0)
        self.bcs[1] = (-3)*offset #ψ₀'(0)
        self.bcs[2] = (-3/2)*(1-self.param[3]-(2*self.param[2]))*(offset**2) #f₀₀(0)
        self.bcs[3] = (-3)*(1-self.param[3]-(2*self.param[2]))*offset #f₀₀'(0)
        self.bcs[4] = offset**2 #γ₂(0)
        self.bcs[5] = 2*offset #γ₂'(0)

    def gamma(self,a,psi):
        """ evaluates the lower incomplete gammma function γ(a,psi)"""
        g = np.where(psi>=0,gammainc(a,psi)*gamma(a),0) #gammainc is standardised gamma function P(a,x)=γ(a,x)/Γ(a)
        return g

    def r_one(self,psi_0):
        """ evaluates R1 """
        if psi_0 >=0:
            r_one = 9*((np.exp(psi_0)*self.gamma(5/2,psi_0))+(psi_0**(3/2)))/(np.exp(self.param[0])*self.gamma(5/2,self.param[0]))
        else:
            r_one = 0
        return r_one

    def poisson(self,r,y):
        """system of first order ODEs of the form dy/dr=f(r,y), where y=[ψ₀,u,f₀₀,g,γ₂,v]"""
        ode = np.zeros(6)
        ode[0] = y[1]     #dψ₀/dr=u
        ode[1] = ((-2*y[1])/r) + (-9)*np.exp(y[0])*self.gamma(5/2,y[0])/(np.exp(self.param[0])*self.gamma(5/2,self.param[0])) #du/dr
        ode[2] = y[3]     #df₀₀/dr=g
        ode[3] = ((-2*y[3])/r) - self.r_one(y[0])*y[2] + (-9)*(1-self.param[3]-2*self.param[2]) #dg/dr
        ode[4] = y[5]     #γ₂/dr=v
        ode[5] = ((-2*y[5])/r) + ((6*y[4])/(r**2)) - self.r_one(y[0])*y[4] #dv/dr
        return ode

    def truncation(self,t,y):
        """returns ψ₀ at each integration step"""
        return y[0]

    def integrate(self):
        """ numerical integration of ODEs """
        sol = solve_ivp(self.poisson, (self.r[0],self.r[-1]), self.bcs, method = 'DOP853', t_eval=self.r, events=(self.truncation))
            #(system of ODEs, integration range, initial values, method, radii at which solution stored, find truncation radius)
        print(sol.message)
        return sol

    def perturbation_potential(self,r):
        """ evaluates required components of the perturbation potential """
        T_two_zero = 3*m.sqrt(m.pi/5)*(2+self.param[3])*(r**2)
        T_two_two = -3*m.sqrt(3*m.pi/5)*self.param[3]*(r**2)
        C_two_zero = 6*m.sqrt(m.pi/5)*(r**2)
        perturbation = np.array([T_two_zero,T_two_two,C_two_zero])
        return perturbation

    def constants(self,sol):
        """ evaluates the constants A₂₀ and A₂₂ """
        r_trunc = sol.t_events[0][0] #truncation radius
        pert_pot = self.perturbation_potential(r_trunc)
        A_two_zero = (-5*(pert_pot[0]+(self.param[2]*pert_pot[2])))/((sol.y_events[0][0][5]*r_trunc)+(3*sol.y_events[0][0][4]))
        A_two_two = -5*pert_pot[1]/((sol.y_events[0][0][5]*r_trunc)+(3*sol.y_events[0][0][4]))
        A = np.array([A_two_zero,A_two_two])
        return A

    def spherical_harmonics(self,theta,phi):
        """ evaluates the spherical harmonics Y20 and Y22, for given theta and phi """
        Y_two_zero = m.sqrt(5/m.pi)*(1/4)*(3*(np.cos(theta)**2)-1)
        Y_two_two = m.sqrt(15/(2*m.pi))*(1/4)*(np.sin(theta)**2)*np.cos(2*phi)
        Y = np.array([Y_two_zero,Y_two_two])
        return Y

    def internal_solution(self,sol,theta,phi):
        """ constructs the internal solution """
        Alm = self.constants(sol)
        Ylm = self.spherical_harmonics(theta,phi)
        internal = sol.y[0] + (sol.y[2] + (Alm[0]*Ylm[0] + Alm[1]*Ylm[1])*sol.y[4])*self.param[1]
        return internal

    def profiles(self,psi):
        """ obtains the dimensionless density and velocity dispersion profiles """
        rho = np.zeros(len(psi))
        sigma = np.zeros(len(psi))
        for i in range(0,len(psi)):
            if psi[i] >= 0:
                rho[i] = np.exp(psi[i])*self.gamma(5/2,psi[i])
                sigma[i] = np.sqrt((2/5)*self.gamma(5/2,psi[i])/self.gamma(7/2,psi[i]))
            else:
                break
        return rho, sigma

    def tidal_radius(self,psi_x,sol):
        """ finds the tidal radius """
        gradient_x = np.gradient(psi_x)
        tidal_radius = np.interp(0,gradient_x,sol.t)
        print("The tidal radius is " + str(tidal_radius))

if __name__ == "__main__":

    args = sys.argv
    if (len(args) != 7):
        print("Usage Python3 Model.py concentration epsilon zeta nu theta phi")
        sys.exit(1)
    concentration = float(args[1])
    epsilon = float(args[2])
    zeta = float(args[3])
    nu = float(args[4])
    theta = m.radians(float(args[5]))
    phi = m.radians(float(args[6]))

    model = Model(concentration,epsilon,zeta,nu)
    solution = model.integrate()
    psi = model.internal_solution(solution,theta,phi)

    rho,sigma = model.profiles(psi)

    psi_x = model.internal_solution(solution,m.pi/2,0)
    model.tidal_radius(psi_x,solution)
