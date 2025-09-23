# The TIRO Code

Python code for constructing the equilibrium dynamical models of tidally perturbed, rotating stellar systems described in the paper _Tidally Perturbed, Rotating Stellar Systems: Asynchronous Equilibria_, available at: .

The required inputs for TIRO.py are: concentration $\Psi$, tidal strength parameter $\epsilon$, asynchronicity parameter $\zeta$, and galactic potential coefficient $\nu$. Values for the truncation and tidal radii are outputted (in dimensionless units), together with plots of the normalised density and velocity dispersion profiles and slices through the equipotential surfaces.

To reproduce Figures 3, 4, 6 and 7, execute the following: \
&ensp; &ensp; python TIRO.py 4 0.0001211 2 3 \
&ensp; &ensp; python TIRO.py 4 0.0000606 2 3

The module Critical.py finds the critical value of the tidal strength parameter $\epsilon_{cr}$ for inputted values of $\Psi$, $\zeta$ and $\nu$.

The module NBody.py takes the same inputs as TIRO.py and converts outputs into N-Body (Hénon) units. The module Physical_Units.py takes two additional inputs from: central density (in $M_\odot/pc^3$), central velocity dispersion (in $km/s$) and total mass (in $M_\odot$). Scale factors $a$, $A$ and $r_0$ are determined and outputted values are given in corresponding physical units.
