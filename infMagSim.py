# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import math
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg') # GUI backend
import matplotlib.pyplot as plt
import IPython.core.display as IPdisp

# <codecell>

import io
import IPython.nbformat.current
def execute_notebook(nbfile):
    with io.open(nbfile) as f:
        nb = IPython.nbformat.current.read(f, 'json')
    ip = get_ipython()
    for cell in nb.worksheets[0].cells:
        if cell.cell_type != 'code':
            continue
        ip.run_cell(cell.input)

# <codecell>

execute_notebook('infMagSim_cython.ipynb')

# <codecell>

def circular_cross_section(grid, t, t_center, v_drift, radius, object_mask):
    # object_mask will give fraction of each cell inside object
    eps = 1.e-5
    radius += eps
    n_points = len(grid)
    n_cells = len(object_mask)
#    if (n_points==n_cells):
#        n_cells -= 1
    z_min = grid[0]
    z_max = grid[n_points-1]
    z_grid_center = (z_max+z_min)/2
    object_index = n_points/2
    z_object = grid[object_index]
    dz = grid[1]-grid[0]
    if (math.fabs(z_object-z_grid_center)>eps):
        print 'object not centered'
    t_enter = t_center - radius/v_drift
    t_leave = t_center + radius/v_drift
    if (t<t_enter):
        width = 0.
        distance_to = (t_enter-t)*v_drift
    elif (t>t_leave):
        width = 0.
        distance_to = (t-t_leave)*v_drift
    else:
        x = (t-t_center)*v_drift
        width = math.sqrt(1-x*x) # Circular cross-section
        distance_to = 0.
#    if (distance_to>0. and enforce_bdy_cond_outside):
#        width = eps
    for j in range(n_cells):
        object_mask[j] = 0.
    if (width>0.):
        z_object_left = z_object - width
        outside_index_left = int((z_object_left-z_min)/dz)
        zeta_left = z_object_left % dz
        if (zeta_left==0.):
            zeta_left += eps
        elif (zeta_left>dz-eps):
            zeta_left = dz-eps
        object_mask[outside_index_left] = 1.-zeta_left/dz

        z_object_right = z_object + width
        outside_index_right = int((z_object_right-z_min)/dz)+1
        zeta_right = dz - (z_object_right % dz)
        if (zeta_right==0.):
            zeta_right += eps
        elif (zeta_right>dz-eps):
            zeta_right = dz-eps
        object_mask[outside_index_right-1] = 1.-zeta_right/dz

        if (width>dz):
            for j in range(outside_index_left+1, outside_index_right-1):
                object_mask[j] = 1.
    return distance_to

# <codecell>

z_min = -25.
z_max = 25.
n_cells = 1000
n_points = n_cells+1
dz = (z_max-z_min)/(n_points-1)
print 'dz =', dz
eps = 1e-4
grid = np.arange(z_min,z_max+eps,dz,dtype=np.float32)

# <codecell>

extra_storage_factor = 1.2
ion_seed = 384
np.random.seed(ion_seed)
n_ions = 5000000
largest_ion_index = [n_ions-1]
ion_storage_length = int(extra_storage_factor*n_ions)
ions = np.zeros([2,ion_storage_length],dtype=np.float32)
ions[0][0:n_ions] = np.random.rand(n_ions)*(z_max-z_min) + z_min # positions
v_th_i = 1
v_d_i = 0
ions[1][0:n_ions] = np.random.randn(n_ions)*v_th_i + v_d_i # velocities
#ions[2][0:n_ions] = np.ones(n_ions,dtype=np.float32) # relative weights
background_ion_density = n_ions/(z_max-z_min)

# List remaining slots in reverse order to prevent memory fragmentation
empty_ion_slots = -np.ones(ion_storage_length,dtype=np.int)
current_empty_ion_slot = [(ion_storage_length-n_ions)-1]
empty_ion_slots[0:(current_empty_ion_slot[0]+1)] = range(ion_storage_length-1,n_ions-1,-1)

fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(8,2))
for ax, data in zip(axes,ions):
    n_bins = n_points;
    ax.hist(data[0:n_ions],bins=n_bins, histtype='step')
filename='data.png'
plt.savefig(filename)
IPdisp.Image(filename=filename)

# <codecell>

electron_seed = ion_seed+523
np.random.seed(electron_seed)
mass_ratio = 1./1836.
n_electrons = n_ions
largest_electron_index = [n_electrons-1]
electron_storage_length = int(n_electrons*extra_storage_factor)
electrons = np.zeros([2,electron_storage_length],dtype=np.float32)
electrons[0][0:n_electrons] = np.random.rand(n_electrons)*(z_max-z_min) + z_min # positions
v_th_e = 1./math.sqrt(mass_ratio)
v_d_e = 0
electrons[1][0:n_electrons] = np.random.randn(n_electrons)*v_th_e + v_d_e # velocities
#electrons[2][0:n_electrons] = np.ones(n_ions,dtype=np.float32) # relative weights
background_electron_density = n_electrons/(z_max-z_min)

# List remaining slots in reverse order to prevent memory fragmentation
empty_electron_slots = -np.ones(electron_storage_length,dtype=np.int)
current_empty_electron_slot = [(electron_storage_length-n_electrons)-1]
empty_electron_slots[0:(current_empty_electron_slot[0]+1)] = range(electron_storage_length-1,n_electrons-1,-1)

fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(8,2))
for ax, data in zip(axes,electrons):
    n_bins = n_points;
    ax.hist(data[0:n_electrons],bins=n_bins, histtype='step')
filename='data.png'
plt.savefig(filename)
IPdisp.Image(filename=filename)

# <codecell>

v_drift = 0.25*v_th_i
debye_length = 0.125
pot_transp_elong = 2.
object_radius = 1.
t_object_center = (1.+2.*pot_transp_elong*debye_length)/v_drift
t = 0.
#t = t_object_center-2.5
dt = 0.1/v_th_e
ion_charge_to_mass = 1
print t_object_center

object_masks = []
potentials = []
ion_densities = []
electron_densities = []
times = []

object_mask = np.zeros_like(grid) # Could make 1 shorter
object_center_mask = np.zeros_like(grid) # Could make 1 shorter
circular_cross_section(grid,1.e-8,1.,1.,1.,object_center_mask)

ion_density = np.zeros_like(grid)
accumulate_density(grid, object_mask, background_ion_density, largest_ion_index, ions, ion_density)
electron_density = np.zeros_like(grid)
accumulate_density(grid, object_mask, background_electron_density, largest_electron_index, electrons, electron_density)

potential = np.zeros_like(grid)
initialize_mover(grid, object_mask, potential, dt, ion_charge_to_mass, largest_ion_index, ions)
electron_charge_to_mass = -1/mass_ratio
initialize_mover(grid, object_mask, potential, dt, electron_charge_to_mass, largest_electron_index, electrons)

# <codecell>

%%time
n_steps = 8000
storage_step = 1
#injection_seed = 8734
#np.random.seed(injection_seed)
for k in range(n_steps):
    dist_to_obj = circular_cross_section(grid, t, t_object_center, v_drift, object_radius, object_mask)
    object_potential = -3.
    if (dist_to_obj>0.):
        fraction_of_obj_pot = math.exp(-dist_to_obj/debye_length)
        object_potential *= math.exp(-dist_to_obj/(pot_transp_elong*debye_length))
        poisson_solve(grid, object_center_mask, ion_density-electron_density, debye_length, potential, \
                          object_potential=object_potential, object_transparency=(1.-fraction_of_obj_pot))
    else:
        fraction_of_obj_pot = 1.
        poisson_solve(grid, object_mask, ion_density-electron_density, debye_length, potential, \
                          object_potential=object_potential, object_transparency=0.)
    if (k%storage_step==0):
        times.append(t)
        copy = np.empty_like(object_mask)
        copy[:] = object_mask
        object_masks.append(copy)
        copy = np.empty_like(ion_density)
        copy[:] = ion_density
        ion_densities.append(copy)
        copy = np.empty_like(electron_density)
        copy[:] = electron_density
        electron_densities.append(copy)
        copy = np.empty_like(potential)
        copy[:] = potential
        potentials.append(copy)
    move_particles(grid, object_mask, potential, dt, ion_charge_to_mass, \
                       background_ion_density, largest_ion_index, ions, ion_density, \
                       empty_ion_slots, current_empty_ion_slot)
    move_particles(grid, object_mask, potential, dt, electron_charge_to_mass, \
                       background_electron_density, largest_electron_index, \
                       electrons, electron_density, empty_electron_slots, current_empty_electron_slot)
    expected_ion_injection = 2*dt*v_th_i/math.sqrt(2*math.pi)*n_ions/(z_max-z_min)
    n_ions_inject = int(expected_ion_injection)
    expected_electron_injection = 2*dt*v_th_e/math.sqrt(2*math.pi)*n_electrons/(z_max-z_min)
    n_electrons_inject = int(expected_electron_injection)
    # If expected injection number is small, need to add randomness to get right average rate
    if (expected_ion_injection-n_ions_inject)>np.random.rand():
        n_ions_inject += 1
    if (expected_ion_injection-n_electrons_inject)>np.random.rand():
        n_electrons_inject += 1
    #print current_empty_ion_slot, current_empty_electron_slot, n_ions_inject, n_electrons_inject
    inject_particles(n_ions_inject, grid, dt, v_th_i, background_ion_density, ions, empty_ion_slots, \
                         current_empty_ion_slot, largest_ion_index, ion_density)
    inject_particles(n_electrons_inject, grid, dt, v_th_e, background_electron_density, electrons, empty_electron_slots, \
                         current_empty_electron_slot, largest_electron_index, electron_density)
    #print current_empty_ion_slot, current_empty_electron_slot, largest_ion_index, largest_electron_index
    t += dt
print times[0], times[len(times)-1]

# <codecell>

times_np = np.array(times, dtype=np.float32)
object_masks_np = np.array(object_masks, dtype=np.float32)
potentials_np = np.array(potentials, dtype=np.float32)
ion_densities_np = np.array(ion_densities, dtype=np.float32)
electron_densities_np = np.array(electron_densities, dtype=np.float32)
filename = 'l'+('%.4f' % debye_length)+'_d'+('%.3f' % v_drift)+'_np'+('%.1e' % n_points)+'_ni'+('%.1e' % n_ions)+'_dt'+('%.1e' % dt)
print filename
np.savez(filename, grid=grid, times=times_np, object_masks=object_masks_np, potentials=potentials_np, \
             ion_densities=ion_densities_np, electron_densities=electron_densities_np)

data_file = np.load(filename+'.npz')
print data_file.files
#print data_file['potentials'][k]

# <codecell>

k = len(times)-1
print times[k]
fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(8,6))
for ax, data in zip(axes.flatten(),[potentials[k], ion_densities[k]-electron_densities[k], \
                              object_masks[k], object_masks[k], ion_densities[k], electron_densities[k]]):
    ax.plot(grid,data)
filename='figures/data.png'
plt.savefig(filename)
IPdisp.Image(filename=filename)

# <codecell>

fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(8,2))
for ax, data in zip(axes,electrons):
    n_bins = n_points;
    occupied_slots = (data==data)
    occupied_slots[empty_electron_slots[0:current_empty_electron_slot[0]+1]] = False
    ax.hist(data[occupied_slots],bins=n_bins, histtype='step')
filename='data.png'
plt.savefig(filename)
IPdisp.Image(filename=filename)

# <codecell>

fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(8,2))
for ax, data in zip(axes,ions):
    n_bins = n_points;
    occupied_slots = (data==data)
    occupied_slots[empty_ion_slots[0:current_empty_ion_slot[0]+1]] = False
    ax.hist(data[occupied_slots],bins=n_bins, histtype='step')
filename='data.png'
plt.savefig(filename)
IPdisp.Image(filename=filename)

# <codecell>

# python: 10.3s
# cython: 9.8s
# cimport and cdef: 13.6s
# dtype and ndim: 452ms
# no inj or plot: 13.5ms
# 100x part: 1.25s (might have been fluke; more like below)
# no bounds check: 973ms

