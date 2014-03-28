# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext cythonmagic

# <codecell>

%%cython -lgsl
cimport cython
import numpy as np
cimport numpy as np
import math
import cython_gsl as gsl
cimport cython_gsl as gsl

#@cython.boundscheck(False) # turn of bounds-checking for this function (no speed-up seen)
#@cython.wraparound(False) # turn of negative indices for this function (no speed-up seen)
def move_particles(np.ndarray[np.float32_t, ndim=1] grid, np.ndarray[np.float32_t, ndim=1] object_mask, \
                       np.ndarray[np.float32_t, ndim=1] potential, \
                       float dt, float charge_to_mass, float background_density, largest_index_list, \
                       np.ndarray[np.float32_t, ndim=2] particles, np.ndarray[np.float32_t, ndim=1] density, \
                       np.ndarray[np.int_t, ndim=1] empty_slots, \
                       current_empty_slot_list, int update_position=True, int periodic_particles=False):
    cdef int largest_index = largest_index_list[0]
    cdef int current_empty_slot = current_empty_slot_list[0]
    cdef int largest_allowed_slot = len(empty_slots)-1
    if (update_position and (len(empty_slots)<1 or current_empty_slot<0)):
        print 'move_particles needs list of empty particle slots'
    cdef int n_points = len(grid)
    cdef float z_min = grid[0]
    cdef float z_max = grid[n_points-1]
    cdef float dz = grid[1]-grid[0]
    cdef float inactive_slot_position_flag = 2*z_max
    cdef float eps=1.e-5
    cdef int j, object_index
    for j in range(n_points):
        density[j] = 0
    cdef int within_bounds, within_bounds_before_move, inside_object
    cdef int left_index
    cdef float electric_field
    cdef float accel
    cdef float fraction_to_left
    cdef float current_potential=0.
    cdef int i
    for i in range(largest_index):
        if periodic_particles:
            within_bounds = particles[0,i]<0.99*inactive_slot_position_flag
            particles[0,i] = (particles[0,i]-z_min) % (z_max-z_min) + z_min
        else:
            within_bounds = (particles[0,i]>z_min+eps and particles[0,i]<z_max-eps)
        within_bounds_before_move = within_bounds
        inside_object = False
        if (within_bounds_before_move):
            left_index = int((particles[0,i]-z_min)/dz)
            if periodic_particles and left_index>n_points-2:
                left_index = 0
            if (left_index<0 or left_index>n_points-2):
                print 'bad left_index:', left_index, z_min, particles[0,i], z_max
            electric_field = -(potential[left_index+1]-potential[left_index])/dz
            accel = charge_to_mass*electric_field
            particles[1,i] += accel*dt
            if (update_position):
                particles[0,i] += particles[1,i]*dt
                if periodic_particles:
                    within_bounds = particles[0,i]<0.99*inactive_slot_position_flag
                    particles[0,i] = (particles[0,i]-z_min) % (z_max-z_min) + z_min
                else:
                    within_bounds = (particles[0,i]>z_min+eps and particles[0,i]<z_max-eps)
                if within_bounds:
                    left_index = int((particles[0,i]-z_min)/dz)
                    if periodic_particles and left_index>n_points-2:
                        left_index = 0
                    if (left_index<0 or left_index>n_points-2):
                        print 'bad left_index:', left_index, z_min, particles[0,i], z_max
                    fraction_to_left = ( particles[0,i] % dz )/dz
                    current_potential = potential[left_index]*fraction_to_left + \
                        potential[left_index+1]*(1.-fraction_to_left)
        if within_bounds:
            if (object_mask[left_index]>0.):
                if (object_mask[left_index+1]>0.):
                    if (particles[0,i]>grid[left_index]+(1.-object_mask[left_index])*dz):
                        inside_object = True
                    else:
                        if (particles[0,i]>grid[left_index]+object_mask[left_index]*dz):
                            inside_object = True
        if within_bounds_before_move:
            if (not within_bounds) or inside_object:
                particles[0,i] = inactive_slot_position_flag
                current_empty_slot += 1
                if (current_empty_slot>largest_allowed_slot):
                    print 'bad current_empty_slot:', current_empty_slot, largest_allowed_slot
                empty_slots[current_empty_slot] = i
            else:
                fraction_to_left = ( particles[0,i] % dz )/dz
                density[left_index] += fraction_to_left/background_density
                density[left_index+1] += (1-fraction_to_left)/background_density
    largest_index_list[0] = largest_index
    current_empty_slot_list[0] = current_empty_slot


def accumulate_density(grid, object_mask, background_density, largest_index,  particles, density, \
                           empty_slots, current_empty_slot_list):
    potential = np.zeros_like(grid)
    move_particles(grid, object_mask, potential, 0, 1, background_density, largest_index, particles, density, \
                       empty_slots, current_empty_slot_list, update_position=False)

def initialize_mover(grid, object_mask, potential, dt, chage_to_mass, largest_index, particles, \
                         empty_slots, current_empty_slot_list, periodic_particles=False):
    density = np.zeros_like(grid)
    move_particles(grid, object_mask, potential, -dt/2, chage_to_mass, 1, largest_index, \
                       particles, density, empty_slots, current_empty_slot_list, update_position=False, \
                       periodic_particles=periodic_particles)

def draw_velocities(uniform_sample,v_th,v_d=0):
    if (v_d!=0):
        print 'drift not implemented'
    scaled_sample = 2.*uniform_sample-1.
    return np.sign(scaled_sample)*v_th*np.sqrt(-np.log(np.fabs(scaled_sample))*2)

def inject_particles(int n_inject, np.ndarray[np.float32_t,ndim=1] grid, float dt, float v_th, float background_density, \
                         uniform_2d_sampler, \
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
    #cdef np.ndarray[np.float32_t,ndim=1] partial_dt = dt*np.random.rand(n_inject).astype(np.float32)
    #cdef np.ndarray[np.float32_t,ndim=1] velocities = draw_velocities(np.random.rand(n_inject).astype(np.float32),v_th)
    uniform_2d_sample = uniform_2d_sampler.get(n_inject).astype(np.float32).T
    cdef np.ndarray[np.float32_t,ndim=1] partial_dt = dt*uniform_2d_sample[0]
    cdef np.ndarray[np.float32_t,ndim=1] velocities = draw_velocities(uniform_2d_sample[1],v_th)
    cdef int l,i
    #for velocity_sign in [-1.,1.]:
    for l in range(n_inject):
        if current_empty_slot<0:
            print 'no empty slots'
        i = empty_slots[current_empty_slot]
        #particles[1,i] = velocity_sign*np.fabs(velocities[l])
        #if (velocity_sign<0.):
        particles[1,i] = velocities[l]
	# TODO: debye length shorter than eps could give problems with below
        if (velocities[l]<0.):
            particles[0,i] = z_max-2.*eps + partial_dt[l]*particles[1,i]
        else:
            particles[0,i] = z_min+2.*eps + partial_dt[l]*particles[1,i]
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

def poisson_solve(np.ndarray[np.float32_t,ndim=1] grid, np.ndarray[np.float32_t,ndim=1] object_mask, \
                      np.ndarray[np.float32_t,ndim=1] charge, float debye_length, \
                      np.ndarray[np.float32_t,ndim=1] potential, float object_potential=-4., \
                      float object_transparency=1., int boltzmann_electrons=False):
    ## if boltzmann_electrons then charge=ion_charge
    cdef float eps = 1.e-5
    cdef int n_points = len(grid)
    cdef float z_min = grid[0]
    cdef float z_max = grid[n_points-1]
    cdef float dz = grid[1]-grid[0]

    cdef np.ndarray[np.float64_t,ndim=1] diagonal = -2*np.ones(n_points-2,dtype=np.float64) # Exclude dirichlet boundaries
    cdef np.ndarray[np.float64_t,ndim=1] lower_diagonal = np.ones_like(diagonal)
    cdef np.ndarray[np.float64_t,ndim=1] upper_diagonal = np.ones_like(diagonal)
    cdef np.ndarray[np.float64_t,ndim=1] right_hand_side = -dz*dz*charge.astype(np.float64)[1:n_points-1]/debye_length/debye_length
    if boltzmann_electrons:
        diagonal -= dz*dz*np.exp(potential[1:n_points-1])/debye_length/debye_length
        right_hand_side += dz*dz*np.exp(potential[1:n_points-1]) \
            *(np.ones_like(diagonal)-potential[1:n_points-1])/debye_length/debye_length

    cdef np.ndarray[np.float64_t,ndim=1] diagonal_obj = -2.*np.ones_like(diagonal)
    cdef np.ndarray[np.float64_t,ndim=1] lower_diagonal_obj = np.ones_like(diagonal)
    cdef np.ndarray[np.float64_t,ndim=1] upper_diagonal_obj = np.ones_like(diagonal)
    cdef np.ndarray[np.float64_t,ndim=1] right_hand_side_obj = -dz*dz*charge.astype(np.float64)[1:n_points-1]/debye_length/debye_length
    if boltzmann_electrons:
        diagonal_obj -= dz*dz*np.exp(potential[1:n_points-1])/debye_length/debye_length
        right_hand_side_obj += dz*dz*np.exp(potential[1:n_points-1]) \
            *(np.ones_like(diagonal_obj)-potential[1:n_points-1])/debye_length/debye_length
        # TODO: check that object solve works properly with Boltzmann electrons

    # TODO: resolve issues when object boundaries fall on grid points
    cdef float zeta_left, zeta_right
    cdef int j, jj
    for j in range(1,n_points-1):
        jj = j -1 # -1 since excluding bdy
        if (object_mask[j]>0.):
            if (object_mask[j]<1.-eps):
                # Partially covered cell
                if (object_mask[j+1]>0.):
                    # left closest point outside
                    zeta_left = (1.-object_mask[j])*dz
                    lower_diagonal_obj[jj-1] = zeta_left/dz # -1 since lower diagonal shifted
                    diagonal_obj[jj] = -1.-zeta_left/dz
                    upper_diagonal_obj[jj] = 0.
                    right_hand_side_obj[jj] *= zeta_left/dz
                    right_hand_side_obj[jj] -= object_potential
                elif (object_mask[j-1]<1.-eps):
                    # Only point inside
                    zeta_left = (1.-object_mask[j-1])*dz
                    zeta_right = (1.-object_mask[j])*dz
                    lower_diagonal_obj[jj-1] = -(1.-zeta_left/dz) # -1 since lower diagonal shifted
                    diagonal_obj[jj] = -(zeta_left+zeta_right)/dz
                    upper_diagonal_obj[jj] = -(1.-zeta_right/dz)
                    right_hand_side_obj[jj] = -2.*object_potential
                elif (object_mask[j-1]>1.-eps):
                    # right closest point inside
                    zeta_right = (1.-object_mask[j])*dz
                    lower_diagonal_obj[jj-1] = 0. # -1 since lower diagonal shifted
                    diagonal_obj[jj] = -2.*zeta_right/dz
                    upper_diagonal_obj[jj] = -2.*(1.-zeta_right/dz)
                    right_hand_side_obj[jj] = -2.*object_potential
                else:
                    print 'bad object_mask1'
            elif (object_mask[j]>1.-eps):
                if (object_mask[j-1]<1.-eps):
                    # left closest point inside
                    zeta_left = (1.-object_mask[j-1])*dz
                    lower_diagonal_obj[jj-1] = -2.*(1.-zeta_left/dz) # -1 since lower diagonal shifted
                    diagonal_obj[jj] = -2.*zeta_left/dz
                    upper_diagonal_obj[jj] = 0.
                    right_hand_side_obj[jj] = -2.*object_potential
                elif (object_mask[j-1]>1.-eps):
                    # Interior point
                    right_hand_side_obj[j] = 0. # Constant derivative inside object (hopefully zero)
                else:
                    print 'bad object_mask2'
            else:
                print 'bad object_mask3'
        elif (object_mask[j-1]>0.):
            # right closest point outside
            zeta_right = (1.-object_mask[j-1])*dz
            if (jj>0):
                lower_diagonal_obj[jj-1] = 0. # -1 since lower diagonal shifted
            diagonal_obj[jj] = -1.-zeta_right/dz
            upper_diagonal_obj[jj] = zeta_right/dz
            right_hand_side_obj[jj] *= zeta_right/dz
            right_hand_side_obj[jj] -= object_potential
    upper_diagonal = upper_diagonal*object_transparency + upper_diagonal_obj*(1.-object_transparency)
    diagonal = diagonal*object_transparency + diagonal_obj*(1.-object_transparency)
    lower_diagonal = lower_diagonal*object_transparency + lower_diagonal_obj*(1.-object_transparency)
    right_hand_side = right_hand_side*object_transparency + right_hand_side_obj*(1.-object_transparency)
    cdef np.ndarray[np.float64_t,ndim=1] result = np.zeros_like(diagonal)
    potential[0] = 0.
    potential[n_points-1] = 0.
    tridiagonal_solve(lower_diagonal, diagonal, upper_diagonal, right_hand_side, result)
    cdef np.ndarray[np.float32_t,ndim=1] result_32 = result.astype(np.float32)
    for j in range(1,n_points-1):
        potential[j] = result_32[j-1]

cdef class sobol_sequencer:
    cdef gsl.gsl_qrng *quasi_random_generator
    #cdef gsl.gsl_rng *quasi_random_generator
    cdef int rand_dim
    def __init__(self, int rand_dim=1):
        self.rand_dim = rand_dim
        cdef gsl.gsl_qrng_type *generator_type = gsl.gsl_qrng_sobol
        #cdef gsl.gsl_qrng_type *generator_type = gsl.gsl_qrng_niederreiter_2
        #cdef gsl.gsl_rng_type *generator_type = gsl.gsl_rng_mt19937
        #cdef gsl.gsl_rng_type *generator_type = gsl.gsl_rng_ranlxs2
        #cdef gsl.gsl_rng_type *generator_type = gsl.gsl_rng_ranlxd2
        self.quasi_random_generator = gsl.gsl_qrng_alloc(generator_type, self.rand_dim)
        #self.quasi_random_generator = gsl.gsl_rng_alloc(generator_type)
    def get(self,n):
        cdef np.ndarray[np.float64_t,ndim=1] next_values = np.empty(self.rand_dim, dtype=np.float64)
        cdef np.ndarray[np.float64_t,ndim=2] sequence = np.empty([n,self.rand_dim], dtype=np.float64)
        cdef int i,j
        for i in range(n):
            gsl.gsl_qrng_get(self.quasi_random_generator, <double *> next_values.data)
            for j in range(self.rand_dim):
                #next_values[j] = gsl.gsl_rng_uniform_pos(self.quasi_random_generator)
                sequence[i][j] = next_values[j]
        return sequence

