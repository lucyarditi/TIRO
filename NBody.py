import argparse
import numpy as np
from scipy.integrate import tplquad
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from TIRO import Model

def mass_integrand(r,theta,phi):
    return model.density(model.global_solution(r,theta,phi))*np.square(r)*np.sin(theta)

def kinetic_integrand(r,theta,phi,omega_squared):
    psi = model.global_solution(r,theta,phi)
    dispersion = np.where(psi>=0,model.velocity_dispersion(psi),0)
    return model.density(psi) * (3*np.square(dispersion) + (9/(4*np.pi))*np.square(r*np.sin(theta))*omega_squared) * np.square(r)*np.sin(theta)
    
def total_mass(r_tidal):
    return 8*tplquad(mass_integrand,0,0.5*np.pi,0,0.5*np.pi,0,r_tidal)[0]

def kinetic_energy(r_tidal):
    omega_squared = 4*np.pi*(model.param[2]+1)*model.param[1]
    return 4*tplquad(kinetic_integrand,0,np.pi/2,0,np.pi/2,0,r_tidal,args=([omega_squared]))[0]

def corotating_kinetic_integrand(r,theta,phi):
    psi = model.global_solution(r,theta,phi)
    dispersion = np.where(psi>=0,model.velocity_dispersion(psi),0)
    return model.density(psi)*np.square(dispersion)*np.square(r)*np.sin(theta)

def corotating_kinetic_energy(r_tidal):
    return 12*tplquad(corotating_kinetic_integrand,0,np.pi/2,0,np.pi/2,0,r_tidal)[0]
    
def potential_integrand(r,theta,phi):
    psi = model.global_solution(r,theta,phi)
    omega_diff = -4*np.pi*model.param[2]*model.param[1]
    big_omega = 4*np.pi*model.param[1]
    a_pert = (9/(8*np.pi)) * ((omega_diff*np.square(r*np.sin(theta))) + big_omega*(np.square(r*np.cos(theta))-(model.param[3]*np.square(r*np.sin(theta)*np.cos(phi)))))
    constants = model.constants()
    alpha = constants[2] + constants[4]*model.param[1]
    return model.density(psi)*(a_pert+alpha-psi)*np.square(r)*np.sin(theta)

def total_potential(r_tidal):
    return 4*tplquad(potential_integrand,0,np.pi/2,0,np.pi/2,0,r_tidal)[0]

def little_a(M,K):
    return 4*K/M

def king_radius(a,M):
    return (4*np.pi*a*model.density(model.param[0]))/(9*M)

def big_a(r0,M):
    return 1/(M*np.power(r0,3))

def rescaled_density(rho,A):
    return rho*A

def rescaled_velocity_dispersion(sigma,a):
    return np.sqrt(np.square(sigma)/a)

def rescaled_length(r,r0):
    return r*r0

if __name__ == "__main__":

    """ Parameters """

    parser = argparse.ArgumentParser()
    parser.add_argument("psi",help="concentration",type=float)
    parser.add_argument("epsilon",help="tidal strength parameter",type=float)
    parser.add_argument("zeta",help="asynchronicity aprameter",type=float)
    parser.add_argument("nu",help="galactic potential coefficient",type=float)
    args = parser.parse_args()

    """ Run Poisson Solver """

    model = Model([args.psi,args.epsilon,args.zeta,args.nu])
    model.integrate()
    model.r_tidal = model.tidal_radius()

    """ Convert to N-Body units """

    M = total_mass(model.r_tidal)
    K = kinetic_energy(model.r_tidal)

    a = little_a(M,K)
    r0 = king_radius(a,M)
    A = big_a(r0,M)

    T = corotating_kinetic_energy(model.r_tidal)
    P = total_potential(model.r_tidal)
    print(f'Q: {T/-P}')

    print("The truncation radius is " + str(np.round(rescaled_length(model.r_trunc,r0),decimals=2)))
    print("The tidal radius is " + str(np.round(rescaled_length(model.r_tidal,r0),decimals=2)))
    print("The central density is " + str(np.round(rescaled_density(model.density(model.param[0]),A),decimals=2)))
    print("The central velocity dispersion is " + str(np.round(rescaled_velocity_dispersion(model.velocity_dispersion(model.param[0]),a),decimals=2)))

    """ Plotting """

    r = np.linspace(0,model.r_tidal,100000)[1:]

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
    r = rescaled_length(r,r0)
    r_k = rescaled_length(r_k,r0)

    line_z, = plt.plot(r,density_z/rho_zero,'deeppink',label=r'$z$')
    line_y, = plt.plot(r,density_y/rho_zero,'forestgreen',label=r'$y$')
    line_x, = plt.plot(r,density_x/rho_zero,'blue',label=r'$x$')
    line_k, = plt.plot(r_k,density_k/rho_zero,'k--',label="King")
    
    plt.yscale('log')
    plt.ylim(10**(-6),2)
    plt.xlim(0,np.ceil(r[np.nonzero((density_x/rho_zero)>10**-6)[0][-1]]*10)/10)
    plt.legend(handles=[line_x,line_y,line_z,line_k],frameon=False,fontsize='medium')
    plt.ylabel(r'$\rho/\rho_0$',labelpad = 4,fontsize = 'x-large')
    plt.xlabel(r'$r$',labelpad = 4,fontsize = 'x-large')
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

    line_z, = plt.plot(r,dispersion_z/sigma_zero,'deeppink',label=r'$z$')
    line_y, = plt.plot(r,dispersion_y/sigma_zero,'forestgreen',label=r'$y$')
    line_x, = plt.plot(r,dispersion_x/sigma_zero,'blue',label=r'$x$')
    line_k, = plt.plot(r_k,dispersion_k/sigma_zero,'k--',label="King")

    plt.ylim(0,1.05)
    plt.xlim(0,)
    plt.legend(handles=[line_x,line_y,line_z,line_k],frameon=False,fontsize='medium')
    plt.ylabel(r'$\sigma/\sigma_0$',labelpad = 4,fontsize = 'x-large')
    plt.xlabel(r'$r$',labelpad = 4,fontsize = 'x-large')
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
    critical_potential = model.global_solution(model.r_tidal,np.pi/2,0) #critical surface
    
    thetas = np.linspace(0,np.pi,1000)
    phis = np.linspace(0,2*np.pi,2000)

    boundary_xy = np.array(list(map(model.boundary,np.full_like(phis,np.pi/2),phis)))
    radii_xy = np.zeros((len(phis),len(potentials)+1))
    for i,p in enumerate(potentials):
        radii_xy[0,i] = brentq(model.equipotential,10**-6,boundary_xy[0],args=(np.pi/2,phis[0],p))
    radii_xy[0,-1] = brentq(model.equipotential,boundary_xy[0],model.r_tidal,args=(np.pi/2,phis[0],critical_potential))
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
    radii_xz[0,-1] = brentq(model.equipotential,boundary_xz[0],model.r_tidal,args=(theta_input[0],0,critical_potential))
    for j,t in enumerate(theta_input[1:]):
        for i,p in enumerate(potentials):
            radii_xz[j+1,i] = fsolve(model.equipotential,radii_xz[j,i],args=(t,phi_input[j+1],p))[0]
        radii_xz[j+1,-1] = fsolve(model.equipotential,radii_xz[j,-1],args=(t,phi_input[j+1],critical_potential))[0]

    phi_input_y = np.concatenate((np.full_like(thetas,np.pi/2),np.full_like(thetas,3*np.pi/2)[1:-1]))
    boundary_yz = np.array(list(map(model.boundary,theta_input,phi_input_y)))
    radii_yz = np.zeros((len(theta_input),len(potentials)+1))
    for i,p in enumerate(potentials):
        radii_yz[0,i] = brentq(model.equipotential,10**-6,boundary_yz[0],args=(theta_input[0],np.pi/2,p))
    radii_yz[0,-1] = brentq(model.equipotential,boundary_yz[0],model.r_tidal,args=(theta_input[0],np.pi/2,critical_potential))
    for j,t in enumerate(theta_input[1:]):
        for i,p in enumerate(potentials):
            radii_yz[j+1,i] = fsolve(model.equipotential,radii_yz[j,i],args=(t,phi_input_y[j+1],p))[0]
        radii_yz[j+1,-1] = fsolve(model.equipotential,radii_yz[j,-1],args=(t,phi_input_y[j+1],critical_potential))[0]

    fig, axes = plt.subplots(1,3,figsize=(15, 5))
    axes_lim = rescaled_length(np.ceil(model.r_tidal+0.5),r0)
    if axes_lim < 1.4:
        major_locator = 0.2
    elif axes_lim < 3:
        major_locator = 0.5
    else:
        major_locator = 1

    #plotting xy slice
    
    x_boundary = rescaled_length(np.multiply(boundary_xy,np.cos(phis)),r0)
    y_boundary = rescaled_length(np.multiply(boundary_xy,np.sin(phis)),r0)
    x = rescaled_length(np.multiply(radii_xy,np.cos(phis[:,np.newaxis])),r0)
    y = rescaled_length(np.multiply(radii_xy,np.sin(phis[:,np.newaxis])),r0)

    axes[0].plot(x_boundary,y_boundary,'k')
    for i in range(len(potentials)):
        axes[0].plot(x[:,i],y[:,i],'k')
    if (rescaled_length(model.r_tidal,r0)-np.max(x_boundary))/axes_lim > 0.02:
        axes[0].plot(x[:,-1],y[:,-1],'k:',dashes=[1,1.7])

    axes[0].set_ylabel(r'$y$',labelpad = 4,fontsize = 'x-large',rotation=0)
    axes[0].set_xlabel(r'$x$',labelpad = 4,fontsize = 'x-large')
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

    x_boundary = rescaled_length(np.multiply(boundary_xz,np.multiply(np.sin(theta_input),np.cos(phi_input))),r0)
    z_boundary = rescaled_length(np.multiply(boundary_xz,np.cos(theta_input)),r0)
    x = rescaled_length(np.multiply(radii_xz,np.multiply(np.sin(theta_input[:,np.newaxis]),np.cos(phi_input[:,np.newaxis]))),r0) #multiplies each column in radii by thetas (Reshaped by np.newaxis)
    z = rescaled_length(np.multiply(radii_xz,np.cos(theta_input[:,np.newaxis])),r0)

    axes[1].plot(x_boundary,z_boundary,'k')
    for i in range(len(potentials)):
        axes[1].plot(x[:,i],z[:,i],'k')
    if (rescaled_length(model.r_tidal,r0)-np.max(x_boundary))/axes_lim > 0.02:
        axes[1].plot(x[:,-1],z[:,-1],'k:',dashes=[1,1.7])

    axes[1].set_ylabel(r'$z$',labelpad = 4,fontsize = 'x-large',rotation=0)
    axes[1].set_xlabel(r'$x$',labelpad = 4,fontsize = 'x-large')
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

    y_boundary = rescaled_length(np.multiply(boundary_yz,np.multiply(np.sin(theta_input),np.sin(phi_input_y))),r0)
    z_boundary =rescaled_length(np.multiply(boundary_yz,np.cos(theta_input)),r0)
    y = rescaled_length(np.multiply(radii_yz,np.multiply(np.sin(theta_input[:,np.newaxis]),np.sin(phi_input_y[:,np.newaxis]))),r0)
    z = rescaled_length(np.multiply(radii_yz,np.cos(theta_input[:,np.newaxis])),r0)

    axes[2].plot(y_boundary,z_boundary,'k')
    for i in range(len(potentials)):
        axes[2].plot(y[:,i],z[:,i],'k')
    if (rescaled_length(model.r_tidal,r0)-np.max(x_boundary))/axes_lim > 0.02:
        axes[2].plot(y[:,-1],z[:,-1],'k:',dashes=[1,1.7])
    
    axes[2].set_ylabel(r'$z$',labelpad = 4,fontsize = 'x-large',rotation=0)
    axes[2].set_xlabel(r'$y$',labelpad = 4,fontsize = 'x-large')
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