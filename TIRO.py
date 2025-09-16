import sys
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve
from scipy.special import gammainc, gamma
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
np.seterr(invalid='ignore',divide='ignore')

class Model(object):

    def __init__(self,parameters):
        self.param = np.array(parameters)
        self.range = (10**-6,200)

        """ central boundary conditions, corrected via Frobenius """
        self.y0 = np.zeros(6)
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
        #print(self.sol.message)
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
        """ locates the tidal radius """
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
    
    def boundary(self,theta,phi):
        """ locates the cluster boundary """
        r_bound = fsolve(self.global_solution,self.r_trunc,args=(theta,phi))[0]
        return r_bound
    
    def equipotential(self,r,theta,phi,potential):
        """ method used to locate equipotential surfaces """
        return self.global_solution(r,theta,phi) - potential
    
    def density(self,potential):
        """ returns the density profile for a given potential profile """
        return np.exp(potential)*self.gamma(5/2,potential)
    
    def velocity_dispersion(self,potential):
        """ returns the velocity dispersion profile for a given potential profile """
        return np.sqrt((2/5)*self.gamma(7/2,potential)/self.gamma(5/2,potential))

if __name__ == "__main__":

    """ Parameters """

    args = sys.argv
    if (len(args) != 5):
        print("Usage python TIRO.py concentration tidal_strength asynchronicity galaxy_coefficient")
        sys.exit(1)
    psi = float(args[1]) #concentration
    epsilon = float(args[2]) #tidal strength parameter
    zeta = float(args[3]) #asynchronicity parameter
    nu = float(args[4]) #galactic potential coefficient

    """ Run Poisson Solver """

    model = Model([psi,epsilon,zeta,nu])
    model.integrate()
    r_tidal = model.tidal_radius()

    print("The truncation radius is " + str(np.round(model.r_trunc,decimals=2)))
    print("The tidal radius is " + str(np.round(r_tidal,decimals=2)))
    

    """ Plotting """

    r = np.linspace(0,r_tidal,100000)[1:]

    potential_x = model.global_solution(r,np.pi/2,0)
    potential_y = model.global_solution(r,np.pi/2,np.pi/2)
    potential_z = model.global_solution(r,0,0)

    density_x = model.density(potential_x)
    density_y = model.density(potential_y)
    density_z = model.density(potential_z)

    dispersion_x = model.velocity_dispersion(potential_x)
    dispersion_y = model.velocity_dispersion(potential_y)
    dispersion_z = model.velocity_dispersion(potential_z)
    
    r_k = np.linspace(0,model.r_trunc,10000)[1:]
    potential_k = model.sol.sol(r_k)[0] #King model potential
    density_k = model.density(potential_k)
    dispersion_k = model.velocity_dispersion(potential_k)

    #plots normalised density profiles

    rho_zero = model.density(model.param[0]) #central dimensionless density

    line_z, = plt.plot(r,density_z/rho_zero,'deeppink',label=r'$\hat{z}$')
    line_y, = plt.plot(r,density_y/rho_zero,'forestgreen',label=r'$\hat{y}$')
    line_x, = plt.plot(r,density_x/rho_zero,'blue',label=r'$\hat{x}$')
    line_k, = plt.plot(r_k,density_k/rho_zero,'k--',label="King")
    
    plt.yscale('log')
    plt.ylim(10**(-6),2)
    plt.xlim(0,np.ceil(r[np.nonzero((density_x/rho_zero)>10**-6)[0][-1]]))
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

    plt.savefig("Density.png")
    plt.show()

    #plots normalised velocity dispersion profiles

    sigma_zero = model.velocity_dispersion(model.param[0]) #central dimensionless velocity dispersion

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

    plt.savefig("Velocity_Dispersion.png")
    plt.show()

    # plots slices through equipotentials

    potentials = np.array([0.0025,0.0125,0.025,0.05,0.125,0.25,0.5,0.75])*model.param[0]
    critical_potential = model.global_solution(r_tidal,np.pi/2,0) #critical surface
    
    thetas = np.linspace(0,np.pi,1000)
    phis = np.linspace(0,2*np.pi,2000)

    boundary_xy = np.array(list(map(model.boundary,np.full_like(phis,np.pi/2),phis)))
    radii_xy = np.zeros((len(phis),len(potentials)+1))
    for i,p in enumerate(potentials):
        radii_xy[0,i] = brentq(model.equipotential,10**-6,boundary_xy[0],args=(np.pi/2,phis[0],p))
    radii_xy[0,-1] = brentq(model.equipotential,boundary_xy[0],r_tidal,args=(np.pi/2,phis[0],critical_potential))
    for j,ph in enumerate(phis[1:]):
        for i,p in enumerate(potentials):
            radii_xy[j+1,i] = fsolve(model.equipotential,radii_xy[j,i],args=(np.pi/2,ph,p))[0]
        radii_xy[j+1,-1] = fsolve(model.equipotential,radii_xy[j,-1],args=(np.pi/2,ph,critical_potential))[0]

    theta_input = np.concatenate((thetas,np.flipud(thetas)[1:-1]))
    phi_input = np.concatenate((np.zeros_like(thetas),np.full_like(thetas,np.pi)[1:-1]))
    boundary_xz = np.array(list(map(model.boundary,theta_input,phi_input)))
    radii_xz = np.zeros((len(theta_input),len(potentials)+1))
    for i,p in enumerate(potentials):
        radii_xz[0,i] = brentq(model.equipotential,10**-6,boundary_xz[0],args=(theta_input[0],0,p))
    radii_xz[0,-1] = brentq(model.equipotential,boundary_xz[0],r_tidal,args=(theta_input[0],0,critical_potential))
    for j,t in enumerate(theta_input[1:]):
        for i,p in enumerate(potentials):
            radii_xz[j+1,i] = fsolve(model.equipotential,radii_xz[j,i],args=(t,phi_input[j+1],p))[0]
        radii_xz[j+1,-1] = fsolve(model.equipotential,radii_xz[j,-1],args=(t,phi_input[j+1],critical_potential))[0]

    phi_input_y = np.concatenate((np.full_like(thetas,np.pi/2),np.full_like(thetas,3*np.pi/2)[1:-1]))
    boundary_yz = np.array(list(map(model.boundary,theta_input,phi_input_y)))
    radii_yz = np.zeros((len(theta_input),len(potentials)+1))
    for i,p in enumerate(potentials):
        radii_yz[0,i] = brentq(model.equipotential,10**-6,boundary_yz[0],args=(theta_input[0],np.pi/2,p))
    radii_yz[0,-1] = brentq(model.equipotential,boundary_yz[0],r_tidal,args=(theta_input[0],np.pi/2,critical_potential))
    for j,t in enumerate(theta_input[1:]):
        for i,p in enumerate(potentials):
            radii_yz[j+1,i] = fsolve(model.equipotential,radii_yz[j,i],args=(t,phi_input_y[j+1],p))[0]
        radii_yz[j+1,-1] = fsolve(model.equipotential,radii_yz[j,-1],args=(t,phi_input_y[j+1],critical_potential))[0]

    fig, axes = plt.subplots(1,3,figsize=(15, 5))
    axes_lim = np.ceil(r_tidal+0.5)
    if axes_lim < 14:
        major_locator = 2
    elif axes_lim < 30:
        major_locator = 5
    else:
        major_locator = 10

    #plotting xy slice
    
    x_boundary = np.multiply(boundary_xy,np.cos(phis))
    y_boundary = np.multiply(boundary_xy,np.sin(phis))
    x = np.multiply(radii_xy,np.cos(phis[:,np.newaxis]))
    y = np.multiply(radii_xy,np.sin(phis[:,np.newaxis]))

    axes[0].plot(x_boundary,y_boundary,'k')
    for i in range(len(potentials)):
        axes[0].plot(x[:,i],y[:,i],'k')
    if (r_tidal-np.max(x_boundary))/axes_lim > 0.02:
        axes[0].plot(x[:,-1],y[:,-1],'k:',dashes=[1,1.7])

    axes[0].set_ylabel(r'$\hat{y}$',labelpad = 4,fontsize = 'x-large',rotation=0)
    axes[0].set_xlabel(r'$\hat{x}$',labelpad = 4,fontsize = 'x-large')
    secax0 = axes[0].twiny()
    axt0 = secax0.xaxis
    secay0 = axes[0].twinx()
    axr0 = secay0.yaxis
    axes[0].set_box_aspect(1)
    axes[0].set_ylim(-axes_lim,axes_lim)
    axes[0].set_xlim(-axes_lim,axes_lim)  
    secax0.set_xlim(-axes_lim,axes_lim)
    secay0.set_ylim(-axes_lim,axes_lim)
    axes[0].yaxis.set_major_locator(MultipleLocator(major_locator))
    axes[0].xaxis.set_major_locator(MultipleLocator(major_locator))
    axt0.set_major_locator(MultipleLocator(major_locator))
    axr0.set_major_locator(MultipleLocator(major_locator))
    axes[0].minorticks_on()
    axes[0].tick_params(which = 'both',direction='in') #inward pointing ticks
    axes[0].tick_params(length = 6)   #sets length of ticks
    axes[0].tick_params(which = 'minor', length = 4)
    axt0.minorticks_on()
    axt0.set_tick_params(which = 'both',direction='in',labelcolor='none') #no tick label
    axt0.set_tick_params(length = 6)
    axt0.set_tick_params(which = 'minor', length = 4)
    axr0.minorticks_on()
    axr0.set_tick_params(which = 'both',direction='in',labelcolor='none')
    axr0.set_tick_params(length = 6)
    axr0.set_tick_params(which = 'minor', length = 4)

    #plotting xz slice

    x_boundary = np.multiply(boundary_xz,np.multiply(np.sin(theta_input),np.cos(phi_input)))
    z_boundary = np.multiply(boundary_xz,np.cos(theta_input))
    x = np.multiply(radii_xz,np.multiply(np.sin(theta_input[:,np.newaxis]),np.cos(phi_input[:,np.newaxis]))) #multiplies each column in radii by thetas (Reshaped by np.newaxis)
    z = np.multiply(radii_xz,np.cos(theta_input[:,np.newaxis]))

    axes[1].plot(x_boundary,z_boundary,'k')
    for i in range(len(potentials)):
        axes[1].plot(x[:,i],z[:,i],'k')
    if (r_tidal-np.max(x_boundary))/axes_lim > 0.02:
        axes[1].plot(x[:,-1],z[:,-1],'k:',dashes=[1,1.7])

    axes[1].set_ylabel(r'$\hat{z}$',labelpad = 4,fontsize = 'x-large',rotation=0)
    axes[1].set_xlabel(r'$\hat{x}$',labelpad = 4,fontsize = 'x-large')
    secax1 = axes[1].twiny()
    axt1 = secax1.xaxis
    secay1 = axes[1].twinx()
    axr1 = secay1.yaxis
    axes[1].set_box_aspect(1)
    axes[1].set_ylim(-axes_lim,axes_lim)
    axes[1].set_xlim(-axes_lim,axes_lim)  
    secax1.set_xlim(-axes_lim,axes_lim)
    secay1.set_ylim(-axes_lim,axes_lim)
    axes[1].yaxis.set_major_locator(MultipleLocator(major_locator))
    axes[1].xaxis.set_major_locator(MultipleLocator(major_locator))
    axt1.set_major_locator(MultipleLocator(major_locator))
    axr1.set_major_locator(MultipleLocator(major_locator))
    axes[1].minorticks_on()
    axes[1].tick_params(which = 'both',direction='in')
    axes[1].tick_params(length = 6)
    axes[1].tick_params(which = 'minor', length = 4)
    axt1.minorticks_on()
    axt1.set_tick_params(which = 'both',direction='in',labelcolor='none')
    axt1.set_tick_params(length = 6)
    axt1.set_tick_params(which = 'minor', length = 4)
    axr1.minorticks_on()
    axr1.set_tick_params(which = 'both',direction='in',labelcolor='none')
    axr1.set_tick_params(length = 6)
    axr1.set_tick_params(which = 'minor', length = 4)

    #plotting yz slice

    y_boundary = np.multiply(boundary_yz,np.multiply(np.sin(theta_input),np.sin(phi_input_y)))
    z_boundary = np.multiply(boundary_yz,np.cos(theta_input))
    y = np.multiply(radii_yz,np.multiply(np.sin(theta_input[:,np.newaxis]),np.sin(phi_input_y[:,np.newaxis])))
    z = np.multiply(radii_yz,np.cos(theta_input[:,np.newaxis]))

    axes[2].plot(y_boundary,z_boundary,'k')
    for i in range(len(potentials)):
        axes[2].plot(y[:,i],z[:,i],'k')
    if (r_tidal-np.max(x_boundary))/axes_lim > 0.02:
        axes[2].plot(y[:,-1],z[:,-1],'k:',dashes=[1,1.7])
    
    axes[2].set_ylabel(r'$\hat{z}$',labelpad = 4,fontsize = 'x-large',rotation=0)
    axes[2].set_xlabel(r'$\hat{y}$',labelpad = 4,fontsize = 'x-large')
    secax2 = axes[2].twiny()
    axt2 = secax2.xaxis
    secay2 = axes[2].twinx()
    axr2 = secay2.yaxis
    axes[2].set_box_aspect(1)
    axes[2].set_ylim(-axes_lim,axes_lim)
    axes[2].set_xlim(-axes_lim,axes_lim)  
    secax2.set_xlim(-axes_lim,axes_lim)
    secay2.set_ylim(-axes_lim,axes_lim)
    axes[2].yaxis.set_major_locator(MultipleLocator(major_locator))
    axes[2].xaxis.set_major_locator(MultipleLocator(major_locator))
    axt2.set_major_locator(MultipleLocator(major_locator))
    axr2.set_major_locator(MultipleLocator(major_locator))
    axes[2].minorticks_on()
    axes[2].tick_params(which = 'both',direction='in')
    axes[2].tick_params(length = 6)
    axes[2].tick_params(which = 'minor', length = 4)
    axt2.minorticks_on()
    axt2.set_tick_params(which = 'both',direction='in',labelcolor='none')
    axt2.set_tick_params(length = 6)
    axt2.set_tick_params(which = 'minor', length = 4)
    axr2.minorticks_on()
    axr2.set_tick_params(which = 'both',direction='in',labelcolor='none')
    axr2.set_tick_params(length = 6)
    axr2.set_tick_params(which = 'minor', length = 4)
   
    plt.tight_layout()
    plt.savefig("Slice.png")
    plt.show()