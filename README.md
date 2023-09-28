# Blast-Wave

This procedure simulates the Taylor-von Neumann-Sedov Blast Wave model. This model refers to a blast wave induced by a
 strong explosion, such as a nuclear bomb or a supernovae. The blast wave was described by a self-similar solution
 independently by G. I. Taylor, John von Neumann and Leonid Sedov during World War II.
 
 Basic assumptions of the model:
     
     - A large amount of energy E is injected at a point into a medium of density rho.
     - Neglect energy losses due to radiation, i.e., E remains constant with time.
     - Self similar expansion of the blast wave, i.e., the shape of the blast wave remains the same in every time
       step. 
     - Ram pressure of the shock front is much bigger than the ambient pressure, i.e., we neglect the ambient pressure.
     
 Relation with astrophysics concerns (supernovae explosions):
     
     - In a supernovae explosion, as the shock propagates, it accelerates particles which provokes radiation. This means
       that this model is only valid in the adiabatic phase of the shock wave evolution.
     - When the accumulated interstellar material in front of the shock wave is significant, it can slow down the shock.
       Therefore, with this model, we can not investigate these phases of a supernovae.
     - In general, it can be a (roughly) good approximation in certain phases of a supernovae explosion.

 Here, the Euler equations are solved by means of a 2D classic numerical hydrodynamics solver. The imposed initial
 conditions are rho = 1, ux = 0, uy = 0 and e = 1e-06. Additionally, in the 36 central cells (6x6 central square of the
 grid) an additional initial energy density is added.
