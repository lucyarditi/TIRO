import argparse
import numpy as np
from astropy import units as u
from astropy.constants import G
from scipy.integrate import tplquad
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from TIRO import Model

def big_a(rho_zero):
    return rho_zero / model.density(model.param[0])

def little_a(sigma_zero):
    return np.square(model.velocity_dispersion(model.param[0])) / np.square(sigma_zero)

def king_radius(rho_zero,sigma_zero): #in parsecs
    return np.sqrt(9/(4*np.pi*little_a(sigma_zero).to(u.s**2/u.m**2)*G*rho_zero.to(u.kg/u.m**3))).to(u.pc)

def mass_integrand(r,theta,phi):
    return model.density(model.global_solution(r,theta,phi))*(r**2)*np.sin(theta)

def total_mass(r_tidal):
    return tplquad(mass_integrand,0,2*np.pi,0,np.pi,10**-6,r_tidal)

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
    parser.add_argument("-r","--rho",help="central density in solar masses per cubic parsec",type=float)
    parser.add_argument("-s","--sigma",help="central velocity dispersion in km/s",type=float)
    parser.add_argument("-M","--mass",help="total mass in solar masses",type=float)
    args = parser.parse_args()

    """ Run Poisson Solver """
            
    model = Model([args.psi,args.epsilon,args.zeta,args.nu])
    model.integrate()
    r_tidal = model.tidal_radius()

    """ Convert to physical units """

    if args.mass is None:
        if args.rho is None or args.sigma is None:
            parser.error("at least two out of --sigma, --rho and --mass required")
        else:
            rho_zero = args.rho * u.M_sun / u.pc**3
            sigma_zero = args.sigma * u.kilometer / u.s
            A = big_a(rho_zero)
            a = little_a(sigma_zero)
            r0 = king_radius(rho_zero,sigma_zero)
    elif args.sigma is None:
        if args.rho is None or args.mass is None:
            parser.error("at least two out of --sigma, --rho and --mass required")
        else:
            rho_zero = args.rho * u.M_sun / u.pc**3
            mass = args.mass * u.M_sun
            A = big_a(rho_zero)
            M = total_mass(r_tidal)[0]
            r0 = np.cbrt(mass/(M*A))
            a = (9/(4*np.pi*np.square(r0.to(u.m))*G*rho_zero.to(u.kg/u.m**3))).to(u.s**2/u.kilometer**2)
    elif args.rho is None:
        if args.sigma is None or args.mass is None:
            parser.error("at least two out of --sigma, --rho and --mass required")
        else:
            sigma_zero = args.sigma * u.kilometer / u.s
            mass = args.mass * u.M_sun
            a = little_a(sigma_zero)
            M = total_mass(r_tidal)[0]
            r0 = ((4*np.pi*a.to(u.s**2/u.m**2)*G*mass.to(u.kg)*model.density(model.param[0]))/(9*M)).to(u.pc)
            A = (mass/(M*np.power(r0,3))).to(u.M_sun/u.pc**3)
    else:
        parser.error("only two out of --sigma, --rho and --mass required")

    print("The truncation radius is " + str(np.round(rescaled_length(model.r_trunc,r0),decimals=2)))
    print("The tidal radius is " + str(np.round(rescaled_length(r_tidal,r0),decimals=2)))
    if args.sigma is None:
        print("The central velocity dispersion is " + str(np.round(rescaled_velocity_dispersion(model.velocity_dispersion(model.param[0]),a),decimals=2)))
    if args.rho is None:
        print("The central density is " + str(np.round(rescaled_density(model.density(model.param[0]),A),decimals=2)))

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
    r = rescaled_length(r,r0)
    r_k = rescaled_length(r_k,r0)

    line_z, = plt.plot(r,density_z/rho_zero,'deeppink',label=r'$z$')
    line_y, = plt.plot(r,density_y/rho_zero,'forestgreen',label=r'$y$')
    line_x, = plt.plot(r,density_x/rho_zero,'blue',label=r'$x$')
    line_k, = plt.plot(r_k,density_k/rho_zero,'k--',label="King")
    
    plt.yscale('log')
    plt.ylim(10**(-6),2)
    plt.xlim(0,np.ceil(r[np.nonzero((density_x/rho_zero)>10**-6)[0][-1]].value))
    plt.legend(handles=[line_x,line_y,line_z,line_k],frameon=False,fontsize='medium')
    plt.ylabel(r'$\rho/\rho_0$',labelpad = 4,fontsize = 'x-large')
    plt.xlabel(r'$r\ [pc]$',labelpad = 4,fontsize = 'x-large')
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
    plt.xlabel(r'$r\ [pc]$',labelpad = 4,fontsize = 'x-large')
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
    axes_lim = rescaled_length(np.ceil(r_tidal+0.5),r0).value
    if axes_lim < 14:
        major_locator = 2
    elif axes_lim < 30:
        major_locator = 5
    else:
        major_locator = 10

    #plotting xy slice
    
    x_boundary = rescaled_length(np.multiply(boundary_xy,np.cos(phis)),r0)
    y_boundary = rescaled_length(np.multiply(boundary_xy,np.sin(phis)),r0)
    x = rescaled_length(np.multiply(radii_xy,np.cos(phis[:,np.newaxis])),r0)
    y = rescaled_length(np.multiply(radii_xy,np.sin(phis[:,np.newaxis])),r0)

    axes[0].plot(x_boundary,y_boundary,'k')
    for i in range(len(potentials)):
        axes[0].plot(x[:,i],y[:,i],'k')
    if (rescaled_length(r_tidal,r0)-np.max(x_boundary)).value/axes_lim > 0.02:
        axes[0].plot(x[:,-1],y[:,-1],'k:',dashes=[1,1.7])

    axes[0].set_ylabel(r'$y\ [pc]$',labelpad = 4,fontsize = 'x-large')
    axes[0].set_xlabel(r'$x\ [pc]$',labelpad = 4,fontsize = 'x-large')
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
    if (rescaled_length(r_tidal,r0)-np.max(x_boundary)).value/axes_lim > 0.02:
        axes[1].plot(x[:,-1],z[:,-1],'k:',dashes=[1,1.7])

    axes[1].set_ylabel(r'$z\ [pc]$',labelpad = 4,fontsize = 'x-large')
    axes[1].set_xlabel(r'$x\ [pc]$',labelpad = 4,fontsize = 'x-large')
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
    if (rescaled_length(r_tidal,r0)-np.max(x_boundary)).value/axes_lim > 0.02:
        axes[2].plot(y[:,-1],z[:,-1],'k:',dashes=[1,1.7])
    
    axes[2].set_ylabel(r'$z\ [pc]$',labelpad = 4,fontsize = 'x-large')
    axes[2].set_xlabel(r'$y\ [pc]$',labelpad = 4,fontsize = 'x-large')
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