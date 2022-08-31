# The TIRO Code

## Overview

The TIRO code performs numerical integration for the construction of dynamical models of tidally perturbed rotating stellar systems. It is the implementation of the paper:


Python3 is required, as are the sys, math, numpy and scipy packages.

## Inputs

The required inputs for the Poisson solver are: the concentration, tidal strength and asynchronicity parameters $(\Psi,\epsilon,\zeta)$, the galactic potential coefficient $\nu$, and the polar and azimuthal angles $(\theta,\phi)$ defining the direction along which the full internal solution is found. The scale constants A and a are required to obtain values for the central density and King radius.

## Outputs
The dimensionless escape energy (internal solution), density and velocity dispersion profiles are outputted to the file TIRO.txt. The inputs $\Psi$, $\epsilon$, $\theta$, $\nu$, $\theta$, $\phi$ are listed in the first line, and from the third line the columns are: radius, escape energy, density, velocity dispersion (all dimensionless). Numerical values for the tidal radius $\hat{r}_\text{T}$, central density $\rho_0$ and King radius $r_0$ are also outputted. 
