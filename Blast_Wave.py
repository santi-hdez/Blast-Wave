#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 21:46:25 2023

@author: santiago
"""

"""
                                                                              
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
 
 Recommended to run with the default values for a nice simulation.
 
 
 Execution: blast_wave(E, Nx, xbeg, xend, Ny, ybeg, yend, t0, tend, ghost, gamma, CFL, viscosity, Xi, bound_cond, squeme, path_save)


 Input:

  E:                 Additional initial energy to be added to the central cells of the grid. It is added as a energy density
                     according to the grid resolution in the 36 central cells of the grid: e0 = (E/36)/(Dx*Dy)
  Nx:                Number of grid cells in the x-direction.
  xbeg:              Initial value for the computational domain in the x-direction. Defines the size of the grid.
  xend:              Ending value for the computational domain in the x-direction. Defines the size of the grid.
  Ny:                Number of grid cells in the y-direction.
  ybeg:              Initial value for the computational domain in the y-direction. Defines the size of the grid.
  yend:              Ending value for the computational domain in the y-direction. Defines the size of the grid.
  t0:                Initial simulation time.
  tend:              Final simulation time.
  ghost:             Number of ghost cells. Important for imposing the boundary conditions.
  gamma:             Adiabatic exponent. Equation of state: p = (gamma-1)*rho*e, where e is the thermal energy.
  CFL:               Courant number. Security factor <1 used for determining the time step by means of the Courant criterion.
  viscosity:         Boolean parameter. If True, imposes an artificial viscosity. If False, the artificial viscosity is 
                     not added. Only can be used with algorithm='classic'. We want to add artificial viscosity when is
                     desirable to increase the diffusivity near a shock front and only there. We add an artificial
                     viscosity term to our balance equations that mimics the real physics and dissipates kinetic energy
                     into heat. The most popular artificial viscosity ansatz for handling shocks and the one used here is 
                     the so-called von Neumann-Richtmyer artificial viscosity. It is a bulk viscosity which acts as an
                     additional pressure.
  Xi:                Artificial viscosity parameter. Is a tuning parameter which specifies over how many grid cells a shock
                     should be spread out. Normally chosen to be of the order of 2 or 3.
  bound_cond:        Type of boundary conditions to be implemented.
                     Options: reflective --- Main advantage is that both mass and energy are globally conserved since no
                                             mass and no energy leave the computational domain, the system is closed.
                                             Major disadvantage is that waves are also reflected back into the computational
                                             domain at closed boundaries which is highly unphysical.
                              periodic --- Useful if you model a real periodic system or if the computational power is not
                                           sufficient to model the whole domain of interest and one can assume that the
                                           expected pattern of the flow have a periodically repeating nature.
                              free_outflow_inflow --- Main advantage is that all waves that are generated in the
                                                      computational domain can leave the grid without being reflected back.
                                                      One major problem with this condition is the following: if the
                                                      velocity in cell 1 gets u1 > 0, then the state in cell 1 will
                                                      determine the influx of material into the domain (same argument holds
                                                      for the boundary at the other side in cell N) and this can eventually
                                                      lead to any arbitrary inflow of matter into the domain.
                              free_outflow --- These boundaries are an attempt to solve the case of arbitrary inflow
                                               in subsonic cases with free outflow/inflow conditions. Obviously, it does
                                               not generate inflow at all, but mass can be leave the computational domain.
                                               In other words, the boundary is a sink.
  squeme:            Advection squeme to be used to compute the fluxes to be applied in the Godunov method. They are written
                     as flux limiters, i.e., for smooth parts of a solution, the squeme do a 2nd order accurate
                     (flux conserved) advection and for regions near a discontinuity the squeme switch to a 1st order
                     (donor cell/upwind) advection.
                     Options: UW --- donor cell / Upwind squeme.
                              LW --- Lax-Wendroff squeme.
                              BW --- Beam-Warming squeme.
                              Fromm --- Fromm squeme.
                              minmod --- minmod squeme.
                              superbee --- superbee squeme.
                              MC --- MC squeme.
                              vanLeer --- van Leer squeme.
  path_save:         Path where the output of the simulation should be saved.


 Output:    
                                            
  It generates a new folder called Output in the indicated path by 'path_save'. Inside the Output folder, five folders
  called Movie1, Movie2, Movie3, Movie4 and Movie5 are created where videos with the obtained simulation are stored. Each
  video corresponds to one important quantity, namely, density, velocity modulus, energy and pressure.                                                    

"""

def blast_wave(E=1.0, Nx=100, xbeg=-1, xend=1, Ny=100, ybeg=-1, yend=1, t0=0, tend=2.0, ghost=1, gamma=1.4, CFL=0.5, viscosity=False, Xi=3.0, bound_cond='reflective', squeme='MC', path_save='/home/santiago/Documentos'):
    
    import numpy as np
    import sys
    
    sys.path.append('/home/santiago/Documentos/SHPython/SHhydro')
    from Classic_Hydro_Solver_2D import classic_hydro_solver_2d
    
    #Make grid in order to define initial conditions:
    
    Dx = (xend-xbeg)/Nx
    Dy = (yend-ybeg)/Ny
    
    def make_grid():
    
        grid_x = np.arange(xbeg-ghost*(Dx/2),xend+(ghost+1)*(Dx/2),Dx)
        grid_y = np.arange(ybeg-ghost*(Dy/2),yend+(ghost+1)*(Dy/2),Dy)
    
        return grid_x, grid_y
    
    grid_x, grid_y = make_grid()
    
    #Initial conditions:
    
    Q1_0 = np.ones((len(grid_x),len(grid_y)))
    Q2x_0 = np.zeros((len(grid_x),len(grid_y)))
    Q2y_0 = np.zeros((len(grid_x),len(grid_y)))
    Q3_0 = 1e-06*np.ones((len(grid_x),len(grid_y)))
    
    #Initial additional energy density according to the grid resolution in the 36 central cells of the grid:
    
    e0 = (E/36)/(Dx*Dy)
    
    Q3_0[int((len(grid_x)/2)-3), int((len(grid_y)/2)-3)] += e0
    Q3_0[int((len(grid_x)/2)-3), int((len(grid_y)/2)-2)] += e0
    Q3_0[int((len(grid_x)/2)-3), int((len(grid_y)/2)-1)] += e0
    Q3_0[int((len(grid_x)/2)-3), int(len(grid_y)/2)] += e0
    Q3_0[int((len(grid_x)/2)-3), int((len(grid_y)/2)+1)] += e0
    Q3_0[int((len(grid_x)/2)-3), int((len(grid_y)/2)+2)] += e0

    Q3_0[int((len(grid_x)/2)-2), int((len(grid_y)/2)-3)] += e0
    Q3_0[int((len(grid_x)/2)-2), int((len(grid_y)/2)-2)] += e0
    Q3_0[int((len(grid_x)/2)-2), int((len(grid_y)/2)-1)] += e0
    Q3_0[int((len(grid_x)/2)-2), int(len(grid_y)/2)] += e0
    Q3_0[int((len(grid_x)/2)-2), int((len(grid_y)/2)+1)] += e0
    Q3_0[int((len(grid_x)/2)-2), int((len(grid_y)/2)+2)] += e0

    Q3_0[int((len(grid_x)/2)-1), int((len(grid_y)/2)-3)] += e0
    Q3_0[int((len(grid_x)/2)-1), int((len(grid_y)/2)-2)] += e0
    Q3_0[int((len(grid_x)/2)-1), int((len(grid_y)/2)-1)] += e0
    Q3_0[int((len(grid_x)/2)-1), int(len(grid_y)/2)] += e0
    Q3_0[int((len(grid_x)/2)-1), int((len(grid_y)/2)+1)] += e0
    Q3_0[int((len(grid_x)/2)-1), int((len(grid_y)/2)+2)] += e0

    Q3_0[int(len(grid_x)/2), int((len(grid_y)/2)-3)] += e0
    Q3_0[int(len(grid_x)/2), int((len(grid_y)/2)-2)] += e0
    Q3_0[int(len(grid_x)/2), int((len(grid_y)/2)-1)] += e0
    Q3_0[int(len(grid_x)/2), int(len(grid_y)/2)] += e0
    Q3_0[int(len(grid_x)/2), int((len(grid_y)/2)+1)] += e0
    Q3_0[int(len(grid_x)/2), int((len(grid_y)/2)+2)] += e0

    Q3_0[int((len(grid_x)/2)+1), int((len(grid_y)/2)-3)] += e0
    Q3_0[int((len(grid_x)/2)+1), int((len(grid_y)/2)-2)] += e0
    Q3_0[int((len(grid_x)/2)+1), int((len(grid_y)/2)-1)] += e0
    Q3_0[int((len(grid_x)/2)+1), int(len(grid_y)/2)] += e0
    Q3_0[int((len(grid_x)/2)+1), int((len(grid_y)/2)+1)] += e0
    Q3_0[int((len(grid_x)/2)+1), int((len(grid_y)/2)+2)] += e0

    Q3_0[int((len(grid_x)/2)+2), int((len(grid_y)/2)-3)] += e0
    Q3_0[int((len(grid_x)/2)+2), int((len(grid_y)/2)-2)] += e0
    Q3_0[int((len(grid_x)/2)+2), int((len(grid_y)/2)-1)] += e0
    Q3_0[int((len(grid_x)/2)+2), int(len(grid_y)/2)] += e0
    Q3_0[int((len(grid_x)/2)+2), int((len(grid_y)/2)+1)] += e0
    Q3_0[int((len(grid_x)/2)+2), int((len(grid_y)/2)+2)] += e0
    
    classic_hydro_solver_2d(Q1_0=Q1_0, Q2x_0=Q2x_0, Q2y_0=Q2y_0, Q3_0=Q3_0, Nx=Nx, xbeg=xbeg, xend=xend, Ny=Ny, ybeg=ybeg, yend=yend, t0=t0, tend=tend, ghost=ghost, gamma=gamma, CFL=CFL, viscosity=viscosity, Xi=Xi, bound_cond=bound_cond, squeme=squeme, path_save=path_save)
    
    return











