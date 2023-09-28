#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:16:44 2023

@author: santiago
"""

"""
                                                                              
 This procedure constitutes a 2D classic numerical hydrodynamics solver. In this approach, we treat the terms including the
 pressure as source terms on the right hand side of the Euler conservation equations. Then, we solve the advection for the 
 conserved quantities rho, rho*ux , rho*uy and rho*e_tot and, at the end of each time step, we add the source
 terms to the momentum and energy conservation equations. 
 In order to define the time step, we take into account that information should not be transformed over more than one grid
 cell during one time step, or we lose important information. Hence, we limit the time step by means of the Courant Criterion:
 choose the fastest possible information velocity in the computational domain and set the time step so small that the
 information cannot cross one grid cell during this step, multiplied with a security factor < 1, which is called the
 Courant number.
 For the 2D implementation, a directional Strang splitting approach is used here, more specifically: we do first 1/2 a
 time step sweep in x-direction, then 1 time step sweep in y-direction and finally another 1/2 time step sweep in x-direction.
 This method has a slightly higher accuracy.
 
 
 Execution: classic_hydro_solver_2d(Q1_0, Q2x_0, Q2y_0, Q3_0, Nx, xbeg, xend, Ny, ybeg, yend, t0, tend, ghost, gamma, CFL, viscosity, Xi, bound_cond, squeme, path_save)


 Input:

  Q1_0:              Initial conditions for the conserved quantity Q1 = rho (density).
  Q2x_0:             Initial conditions for the conserved quantity Q2x = rho*ux (density * x-velocity).
  Q2y_0:             Initial conditions for the conserved quantity Q2y = rho*uy (density * y-velocity).
  Q3_0:              Initial conditions for the conserved quantity Q3 = rho*e_tot (density * total_energy).
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
                     not added. We want to add artificial viscosity when is desirable to increase the diffusivity near
                     a shock front and only there. We add an artificial viscosity term to our balance equations that mimics
                     the real physics and dissipates kinetic energy into heat. The most popular artificial viscosity
                     ansatz for handling shocks and the one used here is the so-called von Neumann-Richtmyer artificial
                     viscosity. It is a bulk viscosity which acts as an additional pressure.
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

def classic_hydro_solver_2d(Q1_0, Q2x_0, Q2y_0, Q3_0, Nx, xbeg, xend, Ny, ybeg, yend, t0, tend, ghost=1, gamma=1.4, CFL=0.5, viscosity=True, Xi=3.0, bound_cond='reflective', squeme='UW', path_save='/home/santiago/Documentos'):

    import numpy as np
    import sys
    import subprocess
    import os
    import shutil
    import matplotlib.pyplot as plt
    
    sys.path.append('/home/santiago/Documentos/SHPython/SHhydro')
    from Advection_Squemes import advection_squemes
    
    File_Names = []
    FileNames1 = []
    FileNames2 = []
    FileNames3 = []
    FileNames4 = []
    
    fpath1 = path_save+'/Output'
    fpath2 = fpath1+'/Movie1'
    fpath3 = fpath1+'/Movie2'
    fpath4 = fpath1+'/Movie3'
    fpath5 = fpath1+'/Movie4'
    
    os.mkdir(fpath1)
    os.mkdir(fpath2)
    os.mkdir(fpath3)
    os.mkdir(fpath4)
    os.mkdir(fpath5)
    
    # Computational domain:
    
    Dx = (xend-xbeg)/Nx
    Dy = (yend-ybeg)/Ny
    
    # Ghost cells:
    
    Nx_tot = Nx+2*ghost
    Ibeg_x = ghost
    Iend_x = Nx+ghost-1
    
    Ny_tot = Ny+2*ghost
    Ibeg_y = ghost
    Iend_y = Ny+ghost-1
    
    #Initial conditions:
    
    Q1_values = np.copy(Q1_0)
    Q2x_values = np.copy(Q2x_0)
    Q2y_values = np.copy(Q2y_0)
    Q3_values = np.copy(Q3_0)
    
    #Get symmetric grid around computational domain:
        
    def make_grid():
    
        grid_x = np.arange(xbeg-ghost*(Dx/2),xend+(ghost+1)*(Dx/2),Dx)
        grid_y = np.arange(ybeg-ghost*(Dy/2),yend+(ghost+1)*(Dy/2),Dy)
    
        return grid_x, grid_y
    
    #Get buffer quantities:
    
    def buffer(Q1_values, Q2x_values, Q2y_values, Q3_values):
        
        Q1_values_buffer = np.copy(Q1_values)
        Q2x_values_buffer = np.copy(Q2x_values)
        Q2y_values_buffer = np.copy(Q2y_values)
        Q3_values_buffer = np.copy(Q3_values)
        
        return Q1_values_buffer, Q2x_values_buffer, Q2y_values_buffer, Q3_values_buffer
    
    #Pressure computation:
        
    def pressure(Q1_values, Q2x_values, Q2y_values, Q3_values, gamma):
        
        ux = Q2x_values/Q1_values
        uy = Q2y_values/Q1_values
        e_tot = Q3_values/Q1_values
        e_kin = (ux**2+uy**2)/2
        e = e_tot-e_kin
        p = (gamma-1)*Q1_values*e
        
        return p
    
    #Interface velocity:
    
    def u_interface(Q1_values, Q2_values, i, j, flag='x-direction'):
        
        if flag == 'x-direction':
        
            u_inter = 1/2*((Q2_values[i, j]/Q1_values[i, j])+(Q2_values[i+1, j]/Q1_values[i+1, j]))
            
        if flag == 'y-direction':
        
            u_inter = 1/2*((Q2_values[i, j]/Q1_values[i, j])+(Q2_values[i, j+1]/Q1_values[i, j+1]))
        
        return u_inter
        
    #Boundary conditions:
    
    def boundary_conditions(values, index, quantity='density', flag1='reflective', flag2='x-direction'):
        
        if flag2 == 'x-direction':
        
            if flag1 == 'reflective':
                
                if quantity == 'density' or quantity == 'energy':
            
                    values[0, index] = values[1, index]
                
                    values[-1, index] = values[-2, index]
                    
                if quantity == 'velocity':
                    
                    values[0, index] = -values[1, index]
                
                    values[-1, index] = -values[-2, index]
                
            elif flag1 == 'periodic':
                
                values[0, index] = values[-2, index]
            
                values[-1, index] = values[1, index]
                
            elif flag1 == 'free_outflow_inflow':
                
                values[0, index] = values[1, index]
            
                values[-1, index] = values[-2, index]
                
            elif flag1 == 'free_outflow':
                
                if quantity == 'density' or quantity == 'energy':
                
                    values[0, index] = values[1, index]
                
                    values[-1, index] = values[-2, index]
                    
                if quantity == 'velocity':
                    
                    values[0, index] = -abs(values[1, index])
                
                    values[-1, index] = -abs(values[-2, index])
                    
        if flag2 == 'y-direction':
        
            if flag1 == 'reflective':
                
                if quantity == 'density' or quantity == 'energy':
            
                    values[index, 0] = values[index, 1]
                
                    values[index, -1] = values[index, -2]
                    
                if quantity == 'velocity':
                    
                    values[index, 0] = -values[index, 1]
                
                    values[index, -1] = -values[index, -2]
                
            elif flag1 == 'periodic':
                
                values[index, 0] = values[index, -2]
            
                values[index, -1] = values[index, 1]
                
            elif flag1 == 'free_outflow_inflow':
                
                values[index, 0] = values[index, 1]
            
                values[index, -1] = values[index, -2]
                
            elif flag1 == 'free_outflow':
                
                if quantity == 'density' or quantity == 'energy':
                
                    values[index, 0] = values[index, 1]
                
                    values[index, -1] = values[index, -2]
                    
                if quantity == 'velocity':
                    
                    values[index, 0] = -abs(values[index, 1])
                
                    values[index, -1] = -abs(values[index, -2])
                
        return values
    
    #Von Neumann-Richtmyer artificial viscosity:
    
    def artificial_viscosity(Q1_values, Q2x_values, Q2y_values, p, Xi, grid_x, Ibeg_x, Iend_x, grid_y, Ibeg_y, Iend_y):
        
        ux = Q2x_values/Q1_values
        uy = Q2y_values/Q1_values
        
        Pi = 0.0
        
        for x in grid_x[Ibeg_x:Iend_x]:
            
            i = np.where(grid_x==x)
            
            for y in grid_y[Ibeg_y:Iend_y]:
                
                j = np.where(grid_y==y)
        
                if ux[i[0]+1, j[0]] <= ux[i[0]-1, j[0]]:
        
                    Pi += (1/4)*(Xi**2)*((ux[i[0]+1, j[0]]-ux[i[0]-1, j[0]])**2)*Q1_values[i[0], j[0]]
                    
                if uy[i[0], j[0]+1] <= uy[i[0], j[0]-1]:
                    
                    Pi += (1/4)*(Xi**2)*((uy[i[0], j[0]+1]-uy[i[0], j[0]-1])**2)*Q1_values[i[0], j[0]]
        
                p[i[0], j[0]] += Pi
            
        return p
    
    #Time step computation (Courant criterion):
    
    def time_step(Q1_values, Q2x_values, Q2y_values, Q3_values, Dx, Dy, gamma, CFL):
        
        ux = Q2x_values/Q1_values
        uy = Q2y_values/Q1_values
        e_tot = Q3_values/Q1_values
        e_kin = (ux**2+uy**2)/2
        e = e_tot-e_kin
        
        cs = np.sqrt(gamma*(gamma-1)*abs(e))
        vmax = np.max([cs+abs(ux),cs+abs(uy)])
        
        if Dx >= Dy:
            
            Dt = CFL*(Dy/vmax)
            
        else:
            
            Dt = CFL*(Dx/vmax)
            
        return Dt
    
    
    # Main loop:
        
    grid_x, grid_y = make_grid()
    
    t = t0
    it = 0.0
        
    while t < tend:
        
        #Time step:
            
        Dt_tmp = time_step(Q1_values, Q2x_values, Q2y_values, Q3_values, Dx, Dy, gamma, CFL)
        
        #1/2 time step sweep in x-direction:
        
        Dt = Dt_tmp/2
        
        #Buffer:
        
        Q1_values_buffer, Q2x_values_buffer, Q2y_values_buffer, Q3_values_buffer = buffer(Q1_values, Q2x_values, Q2y_values, Q3_values)
        
        for y in grid_y[Ibeg_y:Iend_y]:
            
            j = np.where(grid_y==y)
            
            #Boundary conditions:
                
            boundary_conditions(Q1_values, j[0], quantity='density', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2x_values, j[0], quantity='velocity', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2y_values, j[0], quantity='velocity', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q3_values, j[0], quantity='energy', flag1=bound_cond, flag2='x-direction')
        
            for x in grid_x[Ibeg_x:Iend_x]:
                
                i = np.where(grid_x==x)
                
                #Interface velocities:
                    
                u_inter_R = u_interface(Q1_values_buffer, Q2x_values_buffer, i[0], j[0], flag='x-direction')
                u_inter_L = u_interface(Q1_values_buffer, Q2x_values_buffer, i[0]-1, j[0], flag='x-direction')
        
                #Q1 advection:
                    
                Q1_values[i[0], j[0]] = Q1_values_buffer[i[0], j[0]]-(Dt/Dx)*(advection_squemes(Q1_values_buffer, u_inter_R, Dt, Dx, i[0]+1, j[0], flag1=squeme, flag2='2D', flag3='x-direction')-advection_squemes(Q1_values_buffer, u_inter_L, Dt, Dx, i[0], j[0], flag1=squeme, flag2='2D', flag3='x-direction'))
        
                #Q2x advection:
                
                Q2x_values[i[0], j[0]] = Q2x_values_buffer[i[0], j[0]]-(Dt/Dx)*(advection_squemes(Q2x_values_buffer, u_inter_R, Dt, Dx, i[0]+1, j[0], flag1=squeme, flag2='2D', flag3='x-direction')-advection_squemes(Q2x_values_buffer, u_inter_L, Dt, Dx, i[0], j[0], flag1=squeme, flag2='2D', flag3='x-direction'))
                
                #Q2y advection:
                
                Q2y_values[i[0], j[0]] = Q2y_values_buffer[i[0], j[0]]-(Dt/Dx)*(advection_squemes(Q2y_values_buffer, u_inter_R, Dt, Dx, i[0]+1, j[0], flag1=squeme, flag2='2D', flag3='x-direction')-advection_squemes(Q2y_values_buffer, u_inter_L, Dt, Dx, i[0], j[0], flag1=squeme, flag2='2D', flag3='x-direction'))
                
                #Q3 advection:
            
                Q3_values[i[0], j[0]] = Q3_values_buffer[i[0], j[0]]-(Dt/Dx)*(advection_squemes(Q3_values_buffer, u_inter_R, Dt, Dx, i[0]+1, j[0], flag1=squeme, flag2='2D', flag3='x-direction')-advection_squemes(Q3_values_buffer, u_inter_L, Dt, Dx, i[0], j[0], flag1=squeme, flag2='2D', flag3='x-direction'))
    
            #Boundary conditions:
                
            boundary_conditions(Q1_values, j[0], quantity='density', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2x_values, j[0], quantity='velocity', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2y_values, j[0], quantity='velocity', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q3_values, j[0], quantity='energy', flag1=bound_cond, flag2='x-direction')
            
        t += Dt
    
        #1 time step sweep in y-direction:
                
        Dt = Dt_tmp
        
        #Buffer:
        
        Q1_values_buffer, Q2x_values_buffer, Q2y_values_buffer, Q3_values_buffer = buffer(Q1_values, Q2x_values, Q2y_values, Q3_values)
    
        for x in grid_x[Ibeg_x:Iend_x]:
            
            i = np.where(grid_x==x)
            
            #Boundary conditions:
                
            boundary_conditions(Q1_values, i[0], quantity='density', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q2x_values, i[0], quantity='velocity', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q2y_values, i[0], quantity='velocity', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q3_values, i[0], quantity='energy', flag1=bound_cond, flag2='y-direction')
        
            for y in grid_y[Ibeg_y:Iend_y]:
                
                j = np.where(grid_y==y)
                
                #Interface velocities:
                    
                u_inter_R = u_interface(Q1_values_buffer, Q2y_values_buffer, i[0], j[0], flag='y-direction')
                u_inter_L = u_interface(Q1_values_buffer, Q2y_values_buffer, i[0], j[0]-1, flag='y-direction')
                
                #Q1 advection:
                    
                Q1_values[i[0], j[0]] = Q1_values_buffer[i[0], j[0]]-(Dt/Dy)*(advection_squemes(Q1_values_buffer, u_inter_R, Dt, Dy, i[0], j[0]+1, flag1=squeme, flag2='2D', flag3='y-direction')-advection_squemes(Q1_values_buffer, u_inter_L, Dt, Dy, i[0], j[0], flag1=squeme, flag2='2D', flag3='y-direction'))
        
                #Q2x advection:
                
                Q2x_values[i[0], j[0]] = Q2x_values_buffer[i[0], j[0]]-(Dt/Dy)*(advection_squemes(Q2x_values_buffer, u_inter_R, Dt, Dy, i[0], j[0]+1, flag1=squeme, flag2='2D', flag3='y-direction')-advection_squemes(Q2x_values_buffer, u_inter_L, Dt, Dy, i[0], j[0], flag1=squeme, flag2='2D', flag3='y-direction'))
                
                #Q2y advection:
                
                Q2y_values[i[0], j[0]] = Q2y_values_buffer[i[0], j[0]]-(Dt/Dy)*(advection_squemes(Q2y_values_buffer, u_inter_R, Dt, Dy, i[0], j[0]+1, flag1=squeme, flag2='2D', flag3='y-direction')-advection_squemes(Q2y_values_buffer, u_inter_L, Dt, Dy, i[0], j[0], flag1=squeme, flag2='2D', flag3='y-direction'))
                
                #Q3 advection:
            
                Q3_values[i[0], j[0]] = Q3_values_buffer[i[0], j[0]]-(Dt/Dy)*(advection_squemes(Q3_values_buffer, u_inter_R, Dt, Dy, i[0], j[0]+1, flag1=squeme, flag2='2D', flag3='y-direction')-advection_squemes(Q3_values_buffer, u_inter_L, Dt, Dy, i[0], j[0], flag1=squeme, flag2='2D', flag3='y-direction'))
    
            #Boundary conditions:
                
            boundary_conditions(Q1_values, i[0], quantity='density', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q2x_values, i[0], quantity='velocity', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q2y_values, i[0], quantity='velocity', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q3_values, i[0], quantity='energy', flag1=bound_cond,flag2='y-direction')
        
        t += Dt
        
        #1/2 time step sweep in x-direction:
        
        Dt = Dt_tmp/2
        
        #Buffer:
        
        Q1_values_buffer, Q2x_values_buffer, Q2y_values_buffer, Q3_values_buffer = buffer(Q1_values, Q2x_values, Q2y_values, Q3_values)
        
        for y in grid_y[Ibeg_y:Iend_y]:
            
            j = np.where(grid_y==y)
            
            #Boundary conditions:
                
            boundary_conditions(Q1_values, j[0], quantity='density', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2x_values, j[0], quantity='velocity', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2y_values, j[0], quantity='velocity', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q3_values, j[0], quantity='energy', flag1=bound_cond, flag2='x-direction')
        
            for x in grid_x[Ibeg_x:Iend_x]:
                
                i = np.where(grid_x==x)
                
                #Interface velocities:
                    
                u_inter_R = u_interface(Q1_values_buffer, Q2x_values_buffer, i[0], j[0], flag='x-direction')
                u_inter_L = u_interface(Q1_values_buffer, Q2x_values_buffer, i[0]-1, j[0], flag='x-direction')
        
                #Q1 advection:
                
                Q1_values[i[0], j[0]] = Q1_values_buffer[i[0], j[0]]-(Dt/Dx)*(advection_squemes(Q1_values_buffer, u_inter_R, Dt, Dx, i[0]+1, j[0], flag1=squeme, flag2='2D', flag3='x-direction')-advection_squemes(Q1_values_buffer, u_inter_L, Dt, Dx, i[0], j[0], flag1=squeme, flag2='2D', flag3='x-direction'))
        
                #Q2x advection:
                
                Q2x_values[i[0], j[0]] = Q2x_values_buffer[i[0], j[0]]-(Dt/Dx)*(advection_squemes(Q2x_values_buffer, u_inter_R, Dt, Dx, i[0]+1, j[0], flag1=squeme, flag2='2D', flag3='x-direction')-advection_squemes(Q2x_values_buffer, u_inter_L, Dt, Dx, i[0], j[0], flag1=squeme, flag2='2D', flag3='x-direction'))
                
                #Q2y advection:
                
                Q2y_values[i[0], j[0]] = Q2y_values_buffer[i[0], j[0]]-(Dt/Dx)*(advection_squemes(Q2y_values_buffer, u_inter_R, Dt, Dx, i[0]+1, j[0], flag1=squeme, flag2='2D', flag3='x-direction')-advection_squemes(Q2y_values_buffer, u_inter_L, Dt, Dx, i[0], j[0], flag1=squeme, flag2='2D', flag3='x-direction'))
                
                #Q3 advection:
            
                Q3_values[i[0], j[0]] = Q3_values_buffer[i[0], j[0]]-(Dt/Dx)*(advection_squemes(Q3_values_buffer, u_inter_R, Dt, Dx, i[0]+1, j[0], flag1=squeme, flag2='2D', flag3='x-direction')-advection_squemes(Q3_values_buffer, u_inter_L, Dt, Dx, i[0], j[0], flag1=squeme, flag2='2D', flag3='x-direction'))
    
            #Boundary conditions:
                
            boundary_conditions(Q1_values, j[0], quantity='density', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2x_values, j[0], quantity='velocity', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2y_values, j[0], quantity='velocity', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q3_values, j[0], quantity='energy', flag1=bound_cond, flag2='x-direction')
            
        t += Dt
        
        #Buffer:
        
        Q1_values_buffer, Q2x_values_buffer, Q2y_values_buffer, Q3_values_buffer = buffer(Q1_values, Q2x_values, Q2y_values, Q3_values)
        
        #Pressure:
    
        p = pressure(Q1_values, Q2x_values, Q2y_values, Q3_values, gamma)
            
        #Viscosity:
            
        if viscosity == True:
    
            p = artificial_viscosity(Q1_values, Q2x_values, Q2y_values, p, Xi, grid_x, Ibeg_x, Iend_x, grid_y, Ibeg_y, Iend_y)
            
        #Splitting to add source terms in momentum and energy equations:
            
        ux = Q2x_values_buffer/Q1_values_buffer
        uy = Q2y_values_buffer/Q1_values_buffer
    
        for x in grid_x[Ibeg_x:Iend_x]:
            
            i = np.where(grid_x==x)
            
            for y in grid_y[Ibeg_y:Iend_y]:
                
                j = np.where(grid_y==y)
            
                Q2x_values[i[0], j[0]] = Q2x_values_buffer[i[0], j[0]]-(Dt/(2*Dx))*(p[i[0]+1, j[0]]-p[i[0]-1, j[0]])
                
                Q2y_values[i[0], j[0]] = Q2y_values_buffer[i[0], j[0]]-(Dt/(2*Dy))*(p[i[0], j[0]+1]-p[i[0], j[0]-1])
                
                Q3_values[i[0], j[0]] = Q3_values_buffer[i[0], j[0]]-Dt*((p[i[0]+1, j[0]]*ux[i[0]+1, j[0]]-p[i[0]-1, j[0]]*ux[i[0]-1, j[0]])/(2*Dx)+
                                             (p[i[0], j[0]+1]*uy[i[0], j[0]+1]-p[i[0], j[0]-1]*uy[i[0], j[0]-1])/(2*Dy))
        
        #Plot writing:
        
        print("Writing plots at time", t)
        
        fig, ax=plt.subplots()
        ax.set_aspect('equal')
        im=ax.pcolormesh(grid_x,grid_y,Q1_values,cmap='RdYlBu')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('rho at time='+str(round(t,6)))
        plt.colorbar(im, ax=ax)
        filename1 = "Image_rho_{:.7f}.png".format(it)
        FileNames1.append(filename1) 
        fig.savefig(fpath2+'/'+filename1)
        plt.close()
        
        fig, ax=plt.subplots()
        ax.set_aspect('equal')
        im=ax.pcolormesh(grid_x,grid_y,np.sqrt((Q2x_values/Q1_values)**2+(Q2y_values/Q1_values)**2),cmap='RdYlBu')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('|u| at time='+str(round(t,6)))
        plt.colorbar(im, ax=ax)
        filename2 = "Image_u_{:.7f}.png".format(it)
        FileNames2.append(filename2) 
        fig.savefig(fpath3+'/'+filename2)
        plt.close()
        
        fig, ax=plt.subplots()
        ax.set_aspect('equal')
        im=ax.pcolormesh(grid_x,grid_y,Q3_values/Q1_values,cmap='RdYlBu')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('e at time='+str(round(t,6)))
        plt.colorbar(im, ax=ax)
        filename3 = "Image_e_{:.7f}.png".format(it)
        FileNames3.append(filename3) 
        fig.savefig(fpath4+'/'+filename3)
        plt.close()
        
        fig, ax=plt.subplots()
        ax.set_aspect('equal')
        im=ax.pcolormesh(grid_x,grid_y,p,cmap='RdYlBu')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('p at time='+str(round(t,6)))
        plt.colorbar(im, ax=ax)
        filename4 = "Image_p_{:.7f}.png".format(it)
        FileNames4.append(filename4) 
        fig.savefig(fpath5+'/'+filename4)
        plt.close()
        
        it += 0.0000001
        
    #Use ffmpeg to combine the images in a movie:
    
    subprocess.run(['ffmpeg','-framerate','30','-pattern_type','glob','-i',fpath2+"/*.png",'-c:v','libx264','-pix_fmt','yuv420p','movie_rho.mp4'])
    shutil.move('movie_rho.mp4', fpath2+'/movie_rho.mp4')
    
    subprocess.run(['ffmpeg','-framerate','30','-pattern_type','glob','-i',fpath3+"/*.png",'-c:v','libx264','-pix_fmt','yuv420p','movie_u.mp4'])
    shutil.move('movie_u.mp4', fpath3+'/movie_u.mp4')
    
    subprocess.run(['ffmpeg','-framerate','30','-pattern_type','glob','-i',fpath4+"/*.png",'-c:v','libx264','-pix_fmt','yuv420p','movie_e.mp4'])
    shutil.move('movie_e.mp4', fpath4+'/movie_e.mp4')
    
    subprocess.run(['ffmpeg','-framerate','30','-pattern_type','glob','-i',fpath5+"/*.png",'-c:v','libx264','-pix_fmt','yuv420p','movie_p.mp4'])
    shutil.move('movie_p.mp4', fpath5+'/movie_p.mp4')

    
    
    for filename1 in FileNames1:  #Delete the image files
    
        os.remove(fpath2+'/'+filename1)
        
    for filename2 in FileNames2:  #Delete the image files
    
        os.remove(fpath3+'/'+filename2)
        
    for filename3 in FileNames3:  #Delete the image files
    
        os.remove(fpath4+'/'+filename3)
        
    for filename4 in FileNames4:  #Delete the image files
     
        os.remove(fpath5+'/'+filename4)
        
        
    return


        