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

%load_ext cythonmagic

# <codecell>

z_min = -25.
z_max = 25.
n_cells = 100
n_points = n_cells+1
dz = (z_max-z_min)/(n_points-1)
print 'dz =', dz
eps = 1e-4
grid = np.arange(z_min,z_max+eps,dz,dtype=np.float32)

# <codecell>

extra_storage_factor = 1.2
ion_seed = 384
np.random.seed(ion_seed)
n_ions = 1000000
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

%%cython
cimport cython
import numpy as np
cimport numpy as np

#@cython.boundscheck(False) # turn of bounds-checking for this function (no speed-up seen)
#@cython.wraparound(False) # turn of negative indices for this function (no speed-up seen)
def move_particles(np.ndarray[np.float32_t, ndim=1] grid, np.ndarray[np.float32_t, ndim=1] potential, \
                       float dt, float charge_to_mass, float background_density, largest_index_list, \
                       np.ndarray[np.float32_t, ndim=2] particles, np.ndarray[np.float32_t, ndim=1] density, \
                       np.ndarray[np.int_t, ndim=1] empty_slots, \
                       current_empty_slot_list, int update_position=True, float potential_threshold=-3.):
    cdef int largest_index = largest_index_list[0]
    cdef int current_empty_slot = current_empty_slot_list[0]
    cdef int largest_allowed_slot = len(empty_slots)-1
    if (update_position and (len(empty_slots)<1 or current_empty_slot<0)):
        print 'move_particles needs list of empty particle slots'
    cdef int n_points = len(grid)
    cdef float z_min = grid[0]
    cdef float z_max = grid[n_points-1]
    cdef float dz = grid[1]-grid[0]
    cdef float eps=1.e-5
    cdef int j
    for j in range(n_points):
        density[j] = 0
    cdef int within_bounds
    cdef int left_index
    cdef float electric_field
    cdef float accel
    cdef float fraction_to_left
    cdef float current_potential=0.
    cdef int i
    for i in range(largest_index):
        within_bounds = (particles[0,i]>z_min+eps and particles[0,i]<z_max-eps)
        if (within_bounds):
            left_index = int((particles[0,i]-z_min)/dz)
            if (left_index<0 or left_index>n_points-2):
                print 'bad left_index:', left_index, z_min, particles[0,i], z_max
            electric_field = -(potential[left_index+1]-potential[left_index])/dz
            accel = charge_to_mass*electric_field
            particles[1,i] += accel*dt
            if (update_position):
                particles[0,i] += particles[1,i]*dt
                within_bounds = (particles[0,i]>z_min+eps and particles[0,i]<z_max-eps)
                if within_bounds:
                    left_index = int((particles[0,i]-z_min)/dz)
                    if (left_index<0 or left_index>n_points-2):
                        print 'bad left_index:', left_index, z_min, particles[0,i], z_max
                    fraction_to_left = ( particles[0,i] % dz )/dz
                    current_potential = potential[left_index]*fraction_to_left + \
                        potential[left_index+1]*(1.-fraction_to_left)
                if (not within_bounds) or (current_potential<potential_threshold):
                    particles[0,i] = 2*z_max
                    current_empty_slot += 1
                    if (current_empty_slot>largest_allowed_slot):
                        print 'bad current_empty_slot:', current_empty_slot, largest_allowed_slot
                    empty_slots[current_empty_slot] = i
        if (within_bounds):
            fraction_to_left = ( particles[0,i] % dz )/dz
            density[left_index] += fraction_to_left/background_density
            density[left_index+1] += (1-fraction_to_left)/background_density
    largest_index_list[0] = largest_index
    current_empty_slot_list[0] = current_empty_slot


def accumulate_density(grid, background_density, largest_index,  particles, density):
    potential = np.zeros_like(grid)
    move_particles(grid, potential, 0, 1, background_density, largest_index, particles, density, \
                       np.empty(0,dtype=np.int), [-1], update_position=False)

def initialize_mover(grid, potential, dt, chage_to_mass, largest_index, particles):
    density = np.zeros_like(grid)
    move_particles(grid, potential, -dt/2, chage_to_mass, 1, largest_index, \
                       particles, density, np.empty(0,dtype=np.int), [-1], update_position=False)

def draw_positive_velocities(n_draw,v_th,v_d=0):
    if (v_d!=0):
        print 'drift not implemented'
    return v_th*np.sqrt(-np.log(np.random.rand(n_draw).astype(np.float32))*2)

#def draw_negative_velocities(n_draw,v_th,v_d=0):
#    return -draw_positive_velocities(n_draw,v_th,v_d)

def inject_particles(int n_inject, np.ndarray[np.float32_t,ndim=1] grid, float dt, float v_th, float background_density, \
                         np.ndarray[np.float32_t,ndim=2] particles, np.ndarray[np.int_t,ndim=1] empty_slots, \
                         current_empty_slot_list, largest_index_list, np.ndarray[np.float32_t, ndim=1] density):
    cdef int largest_index = largest_index_list[0]
    cdef int current_empty_slot = current_empty_slot_list[0]
    cdef int n_points = len(grid)
    cdef float eps=1.e-5
    cdef float z_min = grid[0]
    cdef float z_max = grid[n_points-1]
    cdef float dz = grid[1]-grid[0]
    cdef float fraction_to_left
    cdef int left_index
    cdef np.ndarray[np.float32_t,ndim=1] partial_dt = dt*np.random.rand(n_inject).astype(np.float32)
    cdef np.ndarray[np.float32_t,ndim=1] velocity_sign = np.sign(np.random.rand(n_inject).astype(np.float32)-0.5)
    cdef np.ndarray[np.float32_t,ndim=1] positive_velocities = draw_positive_velocities(n_inject,v_th)
    cdef int l,i
    for l in range(n_inject):
        if current_empty_slot<0:
            print 'no empty slots'
        i = empty_slots[current_empty_slot]
        particles[1,i] = velocity_sign[l]*positive_velocities[l]
        if (velocity_sign[l]<0):
            particles[0,i] = z_max-eps + partial_dt[l]*particles[1,i]
        else:
            particles[0,i] = z_min+eps + partial_dt[l]*particles[1,i]
        if (empty_slots[current_empty_slot]>largest_index):
            largest_index = empty_slots[current_empty_slot]
        left_index = int((particles[0,i]-z_min)/dz)
        if (left_index<0 or left_index>n_points-2):
            print 'bad left_index:', left_index, z_min, particles[0,i], z_max, particles[1,i], partial_dt[l]
        else:
            fraction_to_left = ( particles[0,i] % dz )/dz
            density[left_index] += fraction_to_left/background_density
            density[left_index+1] += (1-fraction_to_left)/background_density
            empty_slots[current_empty_slot] = -1
            current_empty_slot -= 1
    largest_index_list[0] = largest_index
    current_empty_slot_list[0] = current_empty_slot

# <codecell>

%%cython
cimport cython
import numpy as np
cimport numpy as np
import math

def tridiagonal_solve(np.ndarray[np.float64_t,ndim=1] a, np.ndarray[np.float64_t,ndim=1] b, \
                          np.ndarray[np.float64_t,ndim=1] c, np.ndarray[np.float64_t,ndim=1] d, \
                          np.ndarray[np.float64_t,ndim=1] x): # also modifies b and d
    cdef int n = len(d)
    cdef int j
    for j in range(n-1):
        d[j+1] -= d[j] * a[j] / b[j]
        b[j+1] -= c[j] * a[j] / b[j]
    for j in reversed(range(n-1)):
        d[j] -= d[j+1] * c[j] / b[j+1]
    for j in range(n):
        x[j] = d[j]/b[j]

def poisson_solve(np.ndarray[np.float32_t,ndim=1] grid, np.ndarray[np.float32_t,ndim=1] charge, float debye_length, \
                      np.ndarray[np.float32_t,ndim=1] potential, float t=0, float t_object_center=10, \
                      float v_drift=1., float object_potential=-4., int enforce_bdy_cond_outside=False):
    cdef float eps = 1.e-5
    cdef int n_points = len(grid)
    cdef float z_min = grid[0]
    cdef float z_max = grid[n_points-1]
    cdef float z_center = (z_max+z_min)/2
    cdef float z_object = grid[n_points/2]
    cdef float dz = grid[1]-grid[0]
    if (math.fabs(z_object-z_center)>eps):
        print 'object not centered'

    cdef np.ndarray[np.float64_t,ndim=1] diagonal = -2*np.ones(n_points-2,dtype=np.float64) # Exclude dirichlet boundaries
    cdef np.ndarray[np.float64_t,ndim=1] lower_diagonal = 1*np.ones_like(diagonal)
    cdef np.ndarray[np.float64_t,ndim=1] upper_diagonal = 1*np.ones_like(diagonal)
    cdef np.ndarray[np.float64_t,ndim=1] right_hand_side = -dz*dz*charge.astype(np.float64)[1:n_points-1]/debye_length/debye_length

    # TODO: resolve issues when object boundaries fall on grid points
    cdef float object_radius = 1.+eps
    cdef float t_enter_object = t_object_center - object_radius/v_drift
    cdef float t_leave_object = t_object_center + object_radius/v_drift
    cdef float object_width, dist_to_obj, x
    if (t<t_enter_object):
        object_width = 0.
        dist_to_obj = (t_enter_object-t)*v_drift
    elif (t>t_leave_object):
        object_width = 0.
        dist_to_obj = (t-t_leave_object)*v_drift
    else:
        x = (t-t_object_center)*v_drift
        object_width = math.sqrt(1-x*x) # Circular cross-section
        dist_to_obj = 0.
    if (dist_to_obj>0. and enforce_bdy_cond_outside):
        #object_potential *= math.exp(-dist_to_obj/debye_length)
        object_width = eps

    cdef float z_object_left, z_object_right
    cdef int outside_index_left, outside_index_right
    cdef float zeta_left, zeta_right
    cdef int object_index, j
    if (object_width>0.):
        z_object_left = z_object - object_width
        outside_index_left = int((z_object_left-z_min)/dz) -1 # -1 since excluding bdy
        zeta_left = z_object_left % dz
        if (zeta_left==0.):
            zeta_left += eps
        elif (zeta_left>dz-eps):
            zeta_left = dz-eps

        z_object_right = z_object + object_width
        outside_index_right = int((z_object_right-z_min)/dz)+1 -1 # -1 since excluding bdy
        zeta_right = dz - (z_object_right % dz)
        if (zeta_right==0.):
            zeta_right += eps
        elif (zeta_right>dz-eps):
            zeta_right = dz-eps

        # Closest point outside to left
        lower_diagonal[outside_index_left-1] = zeta_left/dz # -1 since lower diagonal shifted
        diagonal[outside_index_left] = -1. - zeta_left/dz
        upper_diagonal[outside_index_left] = 0.
        right_hand_side[outside_index_left] *= zeta_left/dz
        right_hand_side[outside_index_left] -= object_potential
        
        # Closest point outside to right
        lower_diagonal[outside_index_right-1] = 0. # -1 since lower diagonal shifted
        diagonal[outside_index_right] = -1. - zeta_right/dz
        upper_diagonal[outside_index_right] = zeta_right/dz
        right_hand_side[outside_index_right] *= zeta_right/dz
        right_hand_side[outside_index_right] -= object_potential

        object_index = n_points/2 -1 # -1 since excluding bdy
        if (object_width<dz):
            # Only point inside
            if (outside_index_left!=object_index-1 or outside_index_right!=object_index+1):
                print 'problem with object indices:', outside_index_left, object_index, outside_index_right
            lower_diagonal[object_index-1] = -(1.-zeta_left/dz)/2. # -1 since lower diagonal shifted
            diagonal[object_index] = -(zeta_left+zeta_right)/dz/2.
            upper_diagonal[object_index] = -(1.-zeta_right/dz)/2.
            right_hand_side[object_index] = -object_potential
        else:
            # Closest point inside to left
            lower_diagonal[outside_index_left+1-1] = -(1.-zeta_left/dz) # -1 since lower diagonal shifted
            diagonal[outside_index_left+1] = -zeta_left/dz
            upper_diagonal[outside_index_left+1] = 0.
            right_hand_side[outside_index_left+1] = -object_potential
            
            # Closest point inside to right
            lower_diagonal[outside_index_right-1-1] = 0. # -1 since lower diagonal shifted
            diagonal[outside_index_right-1] = -zeta_right/dz
            upper_diagonal[outside_index_right-1] = -(1.-zeta_right/dz)
            right_hand_side[outside_index_right-1] = -object_potential
            
            # Remaining interior points
            for j in range(outside_index_left+2, outside_index_right-1):
                right_hand_side[j] = 0. # Constant derivative inside object (hopefully zero)
    
    cdef np.ndarray[np.float64_t,ndim=1] result = np.zeros_like(diagonal)
    potential[0] = 0.
    potential[n_points-1] = 0.
    tridiagonal_solve(lower_diagonal, diagonal, upper_diagonal, right_hand_side, result)
    cdef np.ndarray[np.float32_t,ndim=1] result_32 = result.astype(np.float32)
    for j in range(1,n_points-1):
        potential[j] = result_32[j-1]
    return dist_to_obj

# <codecell>

ion_density = np.zeros_like(grid)
accumulate_density(grid, background_ion_density, largest_ion_index, ions, ion_density)
electron_density = np.zeros_like(grid)
accumulate_density(grid, background_electron_density, largest_electron_index, electrons, electron_density)
potential = np.zeros_like(grid)
no_object_potential = np.zeros_like(grid)
dt = 0.1/v_th_e
ion_charge_to_mass = 1
initialize_mover(grid, potential, dt, ion_charge_to_mass, largest_ion_index, ions)
electron_charge_to_mass = -1/mass_ratio
initialize_mover(grid, potential, dt, electron_charge_to_mass, largest_electron_index, electrons)

# <codecell>

v_drift = 0.5*v_th_i
debye_length = 0.5
t_object_center = (1.+2*debye_length)/v_drift
t = 0.
potentials = []
ion_densities = []
electron_densities = []
times = []
print t_object_center

# <codecell>

%%time
n_steps = 40
storage_step = 1
#injection_seed = 8734
#np.random.seed(injection_seed)
for k in range(n_steps):
    dist_to_obj = poisson_solve(grid, ion_density-electron_density, debye_length, potential, \
                                    t=t, t_object_center=t_object_center, v_drift=v_drift, object_potential=-3., \
                                    enforce_bdy_cond_outside=True)
    dump = poisson_solve(grid, ion_density-electron_density, debye_length, no_object_potential)
    fraction_of_obj_pot = math.exp(-dist_to_obj/debye_length)
    #fraction_of_obj_pot = debye_length/(dist_to_obj+debye_length)
    potential = potential*fraction_of_obj_pot + no_object_potential*(1.-fraction_of_obj_pot)
    if (k%storage_step==0):
        times.append(t)
        copy = np.empty_like(ion_density)
        copy[:] = ion_density
        ion_densities.append(copy)
        copy = np.empty_like(electron_density)
        copy[:] = electron_density
        electron_densities.append(copy)
        copy = np.empty_like(potential)
        copy[:] = potential
        potentials.append(copy)
    move_particles(grid, potential, dt, ion_charge_to_mass, background_ion_density, largest_ion_index, ions, ion_density, \
                       empty_ion_slots, current_empty_ion_slot)
    move_particles(grid, potential, dt, electron_charge_to_mass, background_electron_density, largest_electron_index, \
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
potentials_np = np.array(potentials, dtype=np.float32)
ion_densities_np = np.array(ion_densities, dtype=np.float32)
electron_densities_np = np.array(electron_densities, dtype=np.float32)
filename = 'l'+('%.4f' % debye_length)+'_d'+('%.3f' % v_drift)+'_np'+('%.1e' % n_points)+'_ni'+('%.1e' % n_ions)+'_dt'+('%.1e' % dt)
print filename
np.savez(filename,times=times_np,potentials=potentials_np,ion_densities=ion_densities_np,electron_densities=electron_densities_np)

# <codecell>

data_file = np.load(filename+'.npz')
print data_file.files
print data_file['potentials'][k]

# <codecell>

k = len(times)-1
#k = 12
#k = 2075
print times[k]
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(8,2))
for ax, data in zip(axes,[potentials[k], ion_densities[k]-electron_densities[k]]):
    ax.plot(grid,data)
filename='figures/data.png'
plt.savefig(filename)
IPdisp.Image(filename=filename)

# <codecell>

fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(8,2))
for ax, data in zip(axes,[ion_densities[k],electron_densities[k]]):
    ax.plot(grid,data)
filename='figures/data.png'
plt.savefig(filename)
IPdisp.Image(filename=filename)

# <codecell>

fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(8,2))
for ax, data in zip(axes,[potential, ion_density-electron_density]):
    ax.plot(grid,data)
filename='figures/data.png'
plt.savefig(filename)
IPdisp.Image(filename=filename)

# <codecell>

fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(8,2))
for ax, data in zip(axes,[potential, ion_density-electron_density]):
    ax.plot(grid[n_points/2-3:n_points/2+4],data[n_points/2-3:n_points/2+4])
filename='data.png'
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

# <codecell>


# <codecell>


# <codecell>


# <codecell>


