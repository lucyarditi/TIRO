import sys
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gammainc, gamma
from scipy import constants
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

class Model(object):

    def __init__(self,parameters):
        self.param = np.array(parameters)
        self.range = (10**-6,200)

        self.r = np.linspace(self.range[0],100,10**7)  #dimensionless radii at which solution will be recorded

        """ central boundary conditions, corrected using Frobenius """
        self.y0 = np.zeros(6) #array for storing initial conditions, corrected via frobenius method
        self.y0[0] = self.param[0]-((3/2)*(self.range[0]**2))                      #ψ0(0)
        self.y0[1] = (-3)*self.range[0]                                            #ψ0'(0)
        self.y0[2] = (-3/2)*(1-self.param[3]-(2*self.param[2]))*(self.range[0]**2) #f00(0)
        self.y0[3] = (-3)*(1-self.param[3]-(2*self.param[2]))*self.range[0]        #f00'(0)
        self.y0[4] = self.range[0]**2                                              #γ2(0)
        self.y0[5] = 2*self.range[0]                                               #γ2'(0)

    def poisson(self,r,y):
        """dimensionless Poisson equation: system of six first order 0DEs dy/dr=f(r,y) """
        ode = np.zeros(6) # y=[ψ0,u,f00,g,γ2,v], ode = [dψ0/dr,du/dr,df00/dr,dg/dr,dγ2/dr,dv/dr]
        ode[0] = y[1]     
        ode[1] = ((-2*y[1])/r) + (-9)*np.exp(y[0])*self.gamma(5/2,y[0])/(np.exp(self.param[0])*self.gamma(5/2,self.param[0]))
        ode[2] = y[3]
        ode[3] = ((-2*y[3])/r) - self.r_one(y[0])*y[2] + (-9)*(1-self.param[3]-2*self.param[2])
        ode[4] = y[5]
        ode[5] = ((-2*y[5])/r) + ((6*y[4])/(r**2)) - self.r_one(y[0])*y[4]
        return ode

    def gamma(self,a,psi):
        """ evaluates the lower incomplete gammma function γ(a,psi)"""
        return np.where(psi>=0,gammainc(a,psi)*gamma(a),0)

    def r_one(self,psi_zero):
        """ evaluates R1 for ψ0 """
        if psi_zero >=0:
            r_one = 9*((np.exp(psi_zero)*self.gamma(5/2,psi_zero))+(psi_zero**(3/2)))/(np.exp(self.param[0])*self.gamma(5/2,self.param[0]))
        else:
            r_one = 0
        return r_one

    def truncation_radius(self,t,y):
        """method used in self.integrate to find truncation radius (where ψ0=0)"""
        return y[0]

    def integrate(self):
        """ method for numerical integration of ODEs """
        self.sol = solve_ivp(self.poisson, #system of odes
                        self.range, #integration range
                        self.y0, #initial conditions
                        method = 'DOP853', #Runge-Kutta of order 8
                        events=(self.truncation_radius), #finds r at which ψ0=0
                        dense_output=True #interpolates to output continuous solution
                        )
        print(self.sol.message)
        self.r_trunc = self.sol.t_events[0][0] #truncation radius
    
    def internal_solution(self,r,theta,phi):
        """ constructs the internal solution """
        A = self.constants()[0]
        Y = self.spherical_harmonics(theta,phi)
        internal = self.sol.sol(r)[0] + (self.sol.sol(r)[2] + (A[0]*Y[0] + A[1]*Y[1])*self.sol.sol(r)[4])*self.param[1]
        return internal
    
    def external_solution(self,r,theta,phi):
        """ constructs the external solution """
        constants = self.constants()
        Y = self.spherical_harmonics(theta,phi)
        T = self.tidal_harmonics(r)
        C = self.rotational_harmonics(r)
        external_zero = constants[2] - (constants[1]/r)
        external_one = constants[4] - (constants[3]/r) - ((T[0]+(self.param[2]*C[0]))/(2*np.sqrt(np.pi))) - ((constants[5][0]/r**3)+T[1]+(self.param[2]*C[1]))*Y[0] - ((constants[5][1]/r**3)+T[2])*Y[1]
        external = external_zero + (external_one*self.param[1])
        return external
    
    def global_solution(self,r,theta,phi):
        """ patches together the internal and external solutions """
        solution = np.where(r <= self.r_trunc, self.internal_solution(r,theta,phi), self.external_solution(r,theta,phi))
        return solution

    def constants(self):
        """ evaluates constants in internal and external solutions """
        T = self.tidal_harmonics(self.r_trunc)
        C = self.rotational_harmonics(self.r_trunc)
        
        A = np.zeros(2)
        A[0] = -5*(T[1]+(self.param[2]*C[1]))/((self.sol.y_events[0][0][5]*self.r_trunc)+(3*self.sol.y_events[0][0][4]))
        A[1] = -5*T[2]/((self.sol.y_events[0][0][5]*self.r_trunc)+(3*self.sol.y_events[0][0][4]))
        
        lambda_zero = np.square(self.r_trunc)*self.sol.y_events[0][0][1]
        alpha_zero = lambda_zero / self.r_trunc
        lambda_one = self.sol.y_events[0][0][3]*np.square(self.r_trunc) + ((T[0]+(self.param[2]*C[0]))*self.r_trunc)/np.sqrt(np.pi)
        alpha_one = self.sol.y_events[0][0][2] + (self.sol.y_events[0][0][3]*self.r_trunc) + (3*(T[0]+(self.param[2]*C[0])))/(2*np.sqrt(np.pi))
        
        a = np.zeros(2)
        a[0] = -(self.r_trunc**3)*((A[0]*self.sol.y_events[0][0][4])+T[1]+(self.param[2]*C[1]))
        a[1] = -(self.r_trunc**3)*((A[1]*self.sol.y_events[0][0][4])+T[2])

        return A, lambda_zero, alpha_zero, lambda_one, alpha_one, a

    def spherical_harmonics(self,theta,phi):
        """ evaluates real spherical harmonics Y20 and Y22 """
        Y_two_zero = np.sqrt(5/np.pi)*(1/4)*(3*(np.cos(theta)**2)-1)                           #Y20
        Y_two_two = np.sqrt(15/(2*np.pi))*(1/(2*np.sqrt(2)))*(np.sin(theta)**2)*np.cos(2*phi)  #Y22
        return np.array([Y_two_zero,Y_two_two])

    def tidal_harmonics(self,r):
        """ evaluates spherical harmonic coefficients of tidal potential T"""
        T_zero_zero = -3*np.sqrt(np.pi)*(self.param[3]-1)*np.square(r) #T00(r)
        T_two_zero = 3*np.sqrt(np.pi/5)*(2+self.param[3])*np.square(r) #T20(r)
        T_two_two = -3*np.sqrt(3*np.pi/5)*self.param[3]*np.square(r)   #T22(r)
        return np.array([T_zero_zero,T_two_zero,T_two_two])

    def rotational_harmonics(self,r):
        """ evaluates spherical harmonic coefficients of centrifugal potential C"""
        C_zero_zero = -6*np.sqrt(np.pi)*np.square(r) #C00(r)
        C_two_zero = 6*np.sqrt(np.pi/5)*np.square(r) #C20(r)
        return np.array([C_zero_zero,C_two_zero])

    def tidal_radius(self):
        """ locates the tidal radius (zero of self.external_gradient along x-axis) """
        r_tidal = brentq(self.external_gradient,self.r_trunc,self.range[1],args=(np.pi/2,0))
        return r_tidal
    
    def external_gradient(self,r,theta,phi):
        """ constructs the radial derivative of the external solution """
        constants = self.constants()
        Y = self.spherical_harmonics(theta,phi)
        T = self.tidal_harmonics(r)
        C = self.rotational_harmonics(r)
        gradient_zero = constants[1]/np.square(r)
        gradient_one = constants[3]/np.square(r) - ((T[0]+(self.param[2]*C[0]))/(np.sqrt(np.pi)*r)) - (((-3*constants[5][0])/(r**4))+(2*T[1]/r)+(2*self.param[2]*C[1]/r))*Y[0] - (((-3*constants[5][1])/(r**4))+(2*T[2]/r))*Y[1]
        gradient = gradient_zero + (gradient_one*self.param[1])
        return gradient
    
    def density(self,potential):
        """ returns the density prfile for a given potential profile """
        return np.exp(potential)*self.gamma(5/2,potential)
    
    def velocity_dispersion(self,potential):
        """ returns the velocity dispersion profile for a given potential profile """
        return np.sqrt((2/5)*self.gamma(7/2,potential)/self.gamma(5/2,potential))
    
    def profiles(self):
        """ plots dimensionless density and velocity dispersion profiles """

        r = np.linspace(0,self.tidal_radius(),100000)[1:]

        potential_x = self.global_solution(r,np.pi/2,0)
        potential_y = self.global_solution(r,np.pi/2,np.pi/2)
        potential_z = self.global_solution(r,0,0)

        density_x = self.density(potential_x)
        density_y = self.density(potential_y)
        density_z = self.density(potential_z)

        dispersion_x = self.velocity_dispersion(potential_x)
        dispersion_y = self.velocity_dispersion(potential_y)
        dispersion_z = self.velocity_dispersion(potential_z)
        
        r_k = np.linspace(0,self.r_trunc,10000)[1:]
        potential_k = self.sol.sol(r_k)[0] #King model potential
        density_k = self.density(potential_k)
        dispersion_k = self.velocity_dispersion(potential_k)

        rho_zero = self.density(self.param[0]) #central dimensionless density

        line_z, = plt.plot(r,density_z/rho_zero,'deeppink',label=r'$\hat{z}$')
        line_y, = plt.plot(r,density_y/rho_zero,'forestgreen',label=r'$\hat{y}$')
        line_x, = plt.plot(r,density_x/rho_zero,'blue',label=r'$\hat{x}$')
        line_k, = plt.plot(r_k,density_k/rho_zero,'k--',label="King")
        
        plt.yscale('log')
        plt.ylim(10**(-6),2)
        plt.xlim(0,)
        plt.legend(handles=[line_x,line_y,line_z,line_k],frameon=False,fontsize='medium')
        plt.ylabel(r'$\hat{\rho}/\hat{\rho}_0$',labelpad = 4,fontsize = 'x-large')
        plt.xlabel(r'$\hat{r}$',labelpad = 4,fontsize = 'x-large')
        plt.tick_params(which = 'both',direction='in')
        plt.tick_params(length = 6)
        plt.tick_params(which = 'minor', length = 4)
        ax = plt.gca()
        ax.minorticks_on()
        axt = ax.secondary_xaxis('top') 
        axt.tick_params(which = 'both',direction='in',labelcolor='none')
        axt.tick_params(length = 6)
        axt.tick_params(which = 'minor', length = 4)
        axt.minorticks_on()
        axr = ax.secondary_yaxis('right')
        axr.tick_params(which = 'both',direction='in',labelcolor='none')
        axr.tick_params(length = 6)
        axr.tick_params(which = 'minor', length = 4)

        plt.show()

        sigma_zero = self.velocity_dispersion(self.param[0]) #central dimensionless velocity dispersion

        line_z, = plt.plot(r,dispersion_z/sigma_zero,'deeppink',label=r'$\hat{z}$')
        line_y, = plt.plot(r,dispersion_y/sigma_zero,'forestgreen',label=r'$\hat{y}$')
        line_x, = plt.plot(r,dispersion_x/sigma_zero,'blue',label=r'$\hat{x}$')
        line_k, = plt.plot(r_k,dispersion_k/sigma_zero,'k--',label="King")

        plt.ylim(0,1.05)
        plt.xlim(0,)
        plt.legend(handles=[line_x,line_y,line_z,line_k],frameon=False,fontsize='medium')
        plt.ylabel(r'$\hat{\sigma}/\hat{\sigma}_0$',labelpad = 4,fontsize = 'x-large')
        plt.xlabel(r'$\hat{r}$',labelpad = 4,fontsize = 'x-large')
        plt.tick_params(which = 'both',direction='in')
        plt.tick_params(length = 6)
        plt.tick_params(which = 'minor', length = 4)
        ax = plt.gca()
        ax.minorticks_on()
        axt = ax.secondary_xaxis('top') 
        axt.tick_params(which = 'both',direction='in',labelcolor='none')
        axt.tick_params(length = 6)
        axt.tick_params(which = 'minor', length = 4)
        axt.minorticks_on()
        axr = ax.secondary_yaxis('right')
        axr.tick_params(which = 'both',direction='in',labelcolor='none')
        axr.tick_params(length = 6)
        axr.tick_params(which = 'minor', length = 4)
        axr.minorticks_on()

        plt.show()

if __name__ == "__main__":

    """ Parameters """
    args = sys.argv
    if (len(args) != 5):
        print("Usage Python3 TIRO.py concentration tidal_strength asynchronicity galaxy_coefficient")
        sys.exit(1)
    psi = float(args[1]) #concentration
    epsilon = float(args[2]) #tidal strength parameter
    zeta = float(args[3]) #asynchronicity parameter
    nu = float(args[4]) #galactic potential coefficient

    model = Model([psi,epsilon,zeta,nu])
    model.integrate()

    print("The truncation radius is " + str(model.r_trunc))
    print("The tidal radius is " + str(model.tidal_radius()))

    model.profiles()
