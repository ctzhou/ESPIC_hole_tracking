# Top level script for ESPIC.

from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_id = comm.Get_rank()
n_engines = comm.size

import io
import os
import math
import numpy as np
import scipy.signal as signal
if (mpi_id==0):
    import matplotlib as mpl
    mpl.use('Agg') # non-GUI backend
    import matplotlib.pyplot as plt
from scipy.stats import norm
from infMagSim_cython import *

STORAGE_PATH = "/tmp/"
read_ion_dist_from_file = False
smooth_ion_velocities = False
read_electron_dist_from_file = read_ion_dist_from_file
smooth_electron_velocities = smooth_ion_velocities
dist_filename = ''
project_read_particles = False
projection_angle = np.pi*5/16.
create_electron_dimple = True
moving_box_simulation = True # Using a box that tracks the movement of the hole
counter_streaming_ion_beams = False # Simulate counterstreaming ion beams
v_d_1 = 5. # drift velocity of first ion beam
v_d_2 = -5. # drift velocity of second ion beam
periodic_particles = False
periodic_potential = periodic_particles
prescribe_potential = False
time_steps_immobile_ions = 30000 #number of time steps after which ions are immobile
time_steps_immobile_electrons = 0 #number of time steps before which electrons are immobile
prescribed_potential_growth_cycles = 600 
set_background_acceleration = False
start_step_background_acc = 5000
end_step_background_acc = 8000
start_step_background_acc_phase2 = 15000
end_step_background_acc_phase2 = 22000
def prescribed_potential(grid, t, debye_length=1.):
    z_min = grid[0]
    z_max = grid[-1]
   # return min(1.,t/(z_max-z_min))*np.sin(2.*np.pi*grid/(z_max-z_min))
   # return min(1.,0.01*t/(4*np.power(debye_length,2.)))*np.exp(-np.power(grid,2.)/(2*np.power(debye_length,2.)))
    return min(1.,0.01*600*dt/(4*np.power(debye_length,2.)))*np.exp(-np.power(grid,2.)/(2*np.power(debye_length,2.)))

solve_for_electric_field = False
simulate_moon = False
#v_drift_moon = 25.
#moon_radius = 1.
#t_moon_center = moon_radius/v_drift_moon
boltzmann_electrons = False # Don't move electrons (assume Boltzmann)
boltzmann_potential = boltzmann_electrons # Use Boltzmann relation to solve for potential
decouple_ions_from_electrons = False # Use full potential to move electrons even if Boltzmann for ions
if decouple_ions_from_electrons:
    boltzmann_potential = True
quasineutral = False
zero_electric_field_test = False
if quasineutral:
    boltzmann_electrons = True
quiet_start_only = True
quiet_start_and_injection = False
if quiet_start_only:
    quiet_start_and_injection = False
use_quasirandom_numbers = False
use_quasirandom_dimensions_for_parallelism = False
include_object = False
use_pure_c_mover = True
use_pure_c_solver = use_pure_c_mover


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
        width = math.sqrt(max(0,1-x*x)) # Circular cross-section
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

#def electric_field_filter(grid,data_array,L,fine_mesh):
#    Res = []
#    for x in fine_mesh:
       # H = 4*np.tanh((x-grid)/L)/np.cosh((x-grid)/L)**4
#        H = np.tanh((x-grid)/L)/np.cosh((x-grid)/L)
#        Res.append(np.multiply(data_array,H).sum())
#    return Res

z_min = -3.
z_max = 3.
n_cells = 200
n_points = n_cells+1
n_tracking_mesh_points = 1001
eps = 1e-6
grid, dz = np.linspace(z_min,z_max,num=n_points,endpoint=True,retstep=True)
fine_mesh, fine_dz = np.linspace(z_min,z_max,num=n_tracking_mesh_points,endpoint=True,retstep=True)
grid = grid.astype(np.float32)
fine_mesh = fine_mesh.astype(np.float32)
if quasineutral:
    debye_length = dz
else:
    debye_length = 0.125
number_of_debye_lengths_across = int((z_max-z_min)/debye_length)
number_of_uniforming_bins = number_of_debye_lengths_across*4
number_per_injection_batch = 100
if (mpi_id==0):
    print 'dz, debye_length', dz, debye_length

seed = 384+mpi_id*mpi_id*mpi_id
seedgenrand_c(seed) # seed C MT19937 random number generator


#def histogram2d_uniform_grid(X,Y,x_min,x_max,y_min,y_max,n_bins_x,n_bins_y,eps):
#    print 'Called','Histogram2D'
#    step_x = (x_max-x_min)/n_bins_x
#    step_y = (y_max-y_min)/n_bins_y
#    x_hist_edges = np.arange(x_min,x_max+eps,step_x)
#    y_hist_edges = np.arange(y_min,y_max+eps,step_y)
#    histogram_2d = np.zeros((n_bins_x,n_bins_y),dtype=np.int32)
#    data_2d = np.dstack((X,Y))
#    for C in data_2d[0]:
#        n_x = np.floor((C[0]-x_min)/step_x)
#        n_y = np.floor((C[1]-y_min)/step_y)
#        if (n_y>=0 and n_y<=n_bins_y-1) and (n_x>=0 and n_x<=n_bins_x-1):
#            histogram_2d[n_x,n_y] += 1
#    return histogram_2d, x_hist_edges, y_hist_edges

class uniform_2d_sampler_class:
    def __init__(self,seed,low_discrepancy=False,shared_seed=True,rand_dim=1,parallel_dimensions=1,\
		numbers_per_batch=10000000, mpi_id=0):
	self.low_discrepancy = low_discrepancy
	self.shared_seed = shared_seed
	self.rand_dim = rand_dim
	self.parallel_dimensions = parallel_dimensions
	self.total_dim = rand_dim*parallel_dimensions
	self.numbers_per_batch = numbers_per_batch
	self.mpi_id = mpi_id
	self.total_n = 0
	if low_discrepancy:
	    self.low_discrepancy = True
	    #self.sequencer = ghalton.GeneralizedHalton(self.total_dim,seed) # seed doesn't appear to do anything
	    self.sequencer = sobol_sequencer(rand_dim=self.total_dim)
	else:
	    self.low_discrepancy = False
	    np.random.seed(seed)
    def get_n(self,n):
	self.total_n = self.total_n + n
	if self.low_discrepancy:
	    if self.total_n>1e9:
	        print "WARNING: low-discrepancy sequences may start returning bad values."
	    return np.asarray(self.sequencer.get(n))
	else:
	    return np.random.rand(n,self.total_dim)
    def get(self,n):
	n_per_batch = int(self.numbers_per_batch/self.total_dim)
	n_batches = max(1,n/n_per_batch)
	numbers = np.zeros([n,self.rand_dim])
	for i in range(n_batches):
	    if i==n_batches-1:
	        n_to_get = n-n_per_batch*(n_batches-1)
	    else:
		n_to_get = n_per_batch
	    number_batch = self.get_n(n_to_get)
	    for j in range(self.rand_dim):
		numbers[i*n_per_batch:i*n_per_batch+n_to_get,j] = number_batch[:,self.parallel_dimensions*j+self.mpi_id]
	return numbers
    def get_uniform(self,n,n_bins):
	samples = self.get(n)
	samples[:,0] = (np.arange(n)+samples[:,0])/float(n)
	# TODO: avoid each engine having extra samples in the same bins (might not matter since just for velocity smoothing)
	sample_bins = np.arange(0.,float(n_bins),float(n_bins)/float(n)).astype(np.int32)
	number_in_bins = np.bincount(sample_bins) # TODO: assuming here that there is at least one per bin
	bin_ends = np.cumsum(number_in_bins)
	for i in range(1,self.rand_dim):
	    for j in range(n_bins):
		bin_width = 1./number_in_bins[j]
		current_indices = slice(bin_ends[j]-number_in_bins[j],bin_ends[j])
		samples[current_indices,i] = \
		    np.random.permutation((np.arange(number_in_bins[j])+samples[current_indices,i])*bin_width)
	return samples

class injection_sampler_class:
    def __init__(self,sampler,n_per_batch):
	self.sampler = sampler
	self.shared_seed = sampler.shared_seed
	self.n_per_batch = n_per_batch
	self.n_left_in_batch = 0
	self.samples = np.zeros([0,sampler.rand_dim])
    def get(self,n):
	if self.n_left_in_batch<n:
	    n_extra = n - self.n_left_in_batch
	    n_batches = int(math.ceil(float(n_extra)/float(self.n_per_batch)))
	    n_to_draw = n_batches*self.n_per_batch
	    self.n_left_in_batch = n_to_draw - n_extra
	    bin_width = 1./self.n_per_batch
	    samples = self.sampler.get(n_to_draw)
	    for i in range(1,self.sampler.rand_dim):
		for j in range(n_batches):
		    current_indices = slice(j*self.n_per_batch,(j+1)*self.n_per_batch)
		    samples[current_indices,i] = \
			np.random.permutation((np.arange(self.n_per_batch)+samples[current_indices,i])*bin_width)
	    samples = np.concatenate((self.samples,samples))
	else:
	    samples = self.samples
	    self.n_left_in_batch = self.n_left_in_batch - n
	if self.n_left_in_batch>0:
	    self.samples = samples[-self.n_left_in_batch:]
	else:
	    self.samples = np.zeros([0,self.sampler.rand_dim])
	return samples[:n]

if use_quasirandom_numbers:
    if use_quasirandom_dimensions_for_parallelism:
        uniform_2d_sampler = uniform_2d_sampler_class(seed,low_discrepancy=True,rand_dim=2,\
			       shared_seed=False,parallel_dimensions=n_engines,mpi_id=mpi_id)
    else:
        uniform_2d_sampler = uniform_2d_sampler_class(seed,low_discrepancy=True,rand_dim=2,\
			       shared_seed=True,parallel_dimensions=1)
else:
    uniform_2d_sampler = uniform_2d_sampler_class(seed,shared_seed=False,rand_dim=2)

if quiet_start_and_injection:
    injection_sampler = injection_sampler_class(uniform_2d_sampler,number_per_injection_batch)
else:
    injection_sampler = uniform_2d_sampler

sigma = 40. #sigma is the temperature ratio Te/Ti
v_th_i = 1.
v_d_i = 0.
n_ions = 500000
# This is the initial number of ions inside the computation domain
n_ions_infinity = n_ions 
extra_storage_factor = 6
background_ion_density = n_ions_infinity*n_engines*dz/(z_max-z_min)
n_bins = n_points
v_max_i = 20.*v_th_i
v_min_i = -v_max_i
inactive_slot_position_flag = 2*z_max


largest_ion_index = [n_ions-1]
ion_storage_length = int(extra_storage_factor*n_ions)
ions = np.zeros([2,ion_storage_length],dtype=np.float32)
ions_injection_history = np.zeros([1,ion_storage_length],dtype=np.int32)
n_iterations_ions = 0
def expected_particle_injection(n_infinity,v_th,v_d,dt):
    beta = v_d/(np.sqrt(2.)*v_th)
    return 2*n_infinity*v_th/math.sqrt(2.*math.pi)*np.exp(-np.power(beta,2.))*dt+n_infinity*v_d*math.erf(beta)*dt
#ions[0][0:n_ions] = np.random.rand(n_ions)*(z_max-z_min) + z_min # positions
if uniform_2d_sampler.shared_seed:
    for id in range(n_engines):
	sample = np.asarray(uniform_2d_sampler.get(n_ions)).T
	if id==mpi_id:
	    uniform_2d_sample = sample

if counter_streaming_ion_beams:
    n1_ions = n_ions/2
    n2_ions = n_ions-n1_ions
    uniform_2d_sample_1 = uniform_2d_sampler.get_uniform(n1_ions,number_of_uniforming_bins).T
    uniform_2d_sample_2 = uniform_2d_sampler.get_uniform(n2_ions,number_of_uniforming_bins).T
    ions[0][0:n1_ions] = uniform_2d_sample_1[0]*(z_max-z_min) + z_min
    ions[0][n1_ions:n_ions] = uniform_2d_sample_2[0]*(z_max-z_min) + z_min
    ions[1,0:n1_ions] = uniform_2d_sample_1[1] # velocities
    ions[1,n1_ions:n_ions] = uniform_2d_sample_2[1] # velocities
    inverse_gaussian_cdf_in_place(ions[1,0:n_ions]) # velocities
    ions[1,0:n_ions] *= v_th_i/np.sqrt(sigma) # velocities,sigma is the temperature ratio Te/Ti
    ions[1,0:n1_ions] += v_d_1 # 1st ion beam drift velocity
    ions[1,n1_ions:n_ions] += v_d_2 # 2nd ion beam drift velocity

else:
    if quiet_start_and_injection or quiet_start_only and n_iterations_ions <= 1:
        uniform_2d_sample = uniform_2d_sampler.get_uniform(n_ions,number_of_uniforming_bins).T
        n_iterations_ions += 1
    else:
        uniform_2d_sample = uniform_2d_sampler.get(n_ions).T
    ions[0][0:n_ions] = uniform_2d_sample[0]*(z_max-z_min) + z_min # positions
    if (read_ion_dist_from_file):
        dist_file = np.load(dist_filename)
        ion_distribution_function = dist_file['ion_distribution_function']
        ion_v_edges = dist_file['ion_hist_v_edges']
        weights = ion_distribution_function/np.sum(ion_distribution_function)
        # TODO: option to sample bins quasi-randomly?
        ion_sample_bins = np.random.choice(len(weights),size=n_ions,p=weights)
        left_edge_weights = uniform_2d_sample[1]
        ions[1][0:n_ions] = left_edge_weights*ion_v_edges[ion_sample_bins] +\
            (1.-left_edge_weights)*ion_v_edges[ion_sample_bins+1]
    else:
        #ions[1][0:n_ions] = np.random.randn(n_ions)*v_th_i + v_d_i # velocities
        #ions[1][0:n_ions] = norm.ppf(uniform_2d_sample[1])*v_th_i + v_d_i # velocities
        ions[1,0:n_ions] = uniform_2d_sample[1] # velocities
        inverse_gaussian_cdf_in_place(ions[1,0:n_ions]) # velocities
        ions[1,0:n_ions] *= v_th_i/np.sqrt(sigma) # velocities,sigma is the temperature ratio Te/Ti
        ions[1,0:n_ions] += v_d_i # velocities
        #ions[2][0:n_ions] = np.ones(n_ions) # active slots
        #ions[2][0:n_ions] = np.ones(n_ions,dtype=np.float32) # relative weights
if smooth_ion_velocities:
    # TODO: allow quasi-random numbers here?
    ion_v_smoothing_scale = v_th_i/10.
    if read_ion_dist_from_file:
	ion_v_smoothing_scale = ion_v_edges[1]-ion_v_edges[0]
    ions[1][0:n_ions] += np.random.randn(n_ions)*ion_v_smoothing_scale
if project_read_particles:
    if uniform_2d_sampler.shared_seed:
	for id in range(n_engines):
	    sample = np.asarray(uniform_2d_sampler.get(n_ions)).T
	    if id==mpi_id:
		uniform_2d_sample = sample
    elif quiet_start_and_injection:
	uniform_2d_sample = uniform_2d_sampler.get_uniform(n_ions,number_of_uniforming_bins).T
    else:
	uniform_2d_sample = uniform_2d_sampler.get(n_ions).T
    perpendicular_velocities = norm.ppf(uniform_2d_sample[1])*v_th_i
    velocity_magnitudes = np.sqrt(ions[1][0:n_ions]*ions[1][0:n_ions]
				  + perpendicular_velocities*perpendicular_velocities)
    velocity_angles = np.arctan2(perpendicular_velocities, ions[1][0:n_ions])
    ions[1][0:n_ions] = velocity_magnitudes*np.cos(velocity_angles-projection_angle)
ions[0][n_ions:] = inactive_slot_position_flag
# List remaining slots in reverse order to prevent memory fragmentation
empty_ion_slots = -np.ones(ion_storage_length,dtype=np.int32)
current_empty_ion_slot = [(ion_storage_length-n_ions)-1]
empty_ion_slots[0:(current_empty_ion_slot[0]+1)] = range(ion_storage_length-1,n_ions-1,-1)
ion_hist_n_edges = np.arange(z_min,z_max+eps,(z_max-z_min)/n_bins)
ion_hist_v_edges = np.arange(v_min_i,v_max_i+eps,(v_max_i-v_min_i)/n_bins)
ion_hist_n, ion_hist_n_edges = np.histogram(ions[0][0:n_ions], bins=ion_hist_n_edges)
ion_hist_v, ion_hist_v_edges = np.histogram(ions[1][0:n_ions], bins=ion_hist_v_edges)
comm.Allreduce(MPI.IN_PLACE, ion_hist_n, op=MPI.SUM)
comm.Allreduce(MPI.IN_PLACE, ion_hist_v, op=MPI.SUM)
if (mpi_id==0):
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(8,2))
    for ax, data, bins in zip(axes,[ion_hist_n,ion_hist_v], [ion_hist_n_edges, ion_hist_v_edges]):
	ax.step(bins,np.append(data,[0]), where='post')
    filename='initialIonDistribution.png'
    plt.savefig(filename)


#if not boltzmann_electrons:
mass_ratio = 1./1836.
n_electrons = n_ions
n_electrons_infinity = n_ions_infinity
v_th_e = 1./math.sqrt(mass_ratio)
v_d_e = 0.
background_electron_density = n_electrons_infinity*n_engines*dz/(z_max-z_min)
n_bins = n_points;
v_max_e = 4.*v_th_e
v_min_e = -v_max_e
dimple_velocity_width = v_th_e/15
dimple_velocity = 1.
dimple_height = 0.9
dimple_spatial_width = 4*debye_length
def dimple(v, x, mu=dimple_velocity, sig=dimple_velocity_width, height=dimple_height, Lambda=dimple_spatial_width):
    return height*np.exp( -np.power(v-mu,2.) / (2*np.power(sig,2.)) )*np.exp(-np.power(x,2.) / (2*np.power(Lambda,2.))) 


electron_hist_n_edges = np.arange(z_min,z_max+eps,(z_max-z_min)/n_bins)
electron_hist_v_edges = np.arange(v_min_e,v_max_e+eps,(v_max_e-v_min_e)/n_bins)
if not boltzmann_electrons:
    largest_electron_index = [n_electrons-1]
    electron_storage_length = int(n_electrons*extra_storage_factor)
    electrons = np.zeros([2,electron_storage_length],dtype=np.float32) # Partile data array for qqelectrons
    electrons_injection_history = np.zeros([1,electron_storage_length],dtype=np.int32) # Electron tagging
    n_samples = n_electrons
    n_electrons_so_far = 0
    initialize_z = True
    n_iterations_electrons = 0
    while n_electrons_so_far<n_electrons:
	if (n_samples==n_electrons): # First iteration attempt to initialize all electrons
	    remaining_indices = np.arange(n_electrons)
	#electrons[0][remaining_indices] = Np.Random.Rand(N_Samples)*(Z_Max-Z_Min) + Z_Min # positions
	if uniform_2d_sampler.shared_seed:
	    for id in range(n_engines):
		sample = np.asarray(uniform_2d_sampler.get(n_samples)).T
		if id==mpi_id:
		    uniform_2d_sample = sample
	elif quiet_start_and_injection or quiet_start_only and n_iterations_electrons <= 0:
	    uniform_2d_sample = uniform_2d_sampler.get_uniform(n_samples,number_of_uniforming_bins).T
            n_iterations_electrons += 1
	else:
	    uniform_2d_sample = uniform_2d_sampler.get(n_samples).T
        if initialize_z:
            electrons[0,remaining_indices] = uniform_2d_sample[0]*(z_max-z_min) + z_min # positions
	if (read_electron_dist_from_file):
	    dist_file = np.load(dist_filename)
	    electron_distribution_function = dist_file['electron_distribution_function']
	    electron_v_edges = dist_file['electron_hist_v_edges']
	    weights = electron_distribution_function/np.sum(electron_distribution_function)
	    # TODO: option to sample bins quasi-randomly?
	    electron_sample_bins = np.random.choice(len(weights),size=n_samples,p=weights)
	    left_edge_weights = uniform_2d_sample[1]
	    electrons[1,remaining_indices] = left_edge_weights*electron_v_edges[electron_sample_bins] +\
		(1.-left_edge_weights)*electron_v_edges[electron_sample_bins+1]
	else:
	    #electrons[1,remaining_indices] = np.random.randn(n_samples)*v_th_e + v_d_e # velocities
	    electrons[1,remaining_indices] = norm.ppf(uniform_2d_sample[1]) # velocities
	    #electrons[1,remaining_indices] = uniform_2d_sample[1] # velocities
	    #inverse_gaussian_cdf_in_place(electrons[1,remaining_indices]) # velocities; doesn't work with dimple
	    electrons[1,remaining_indices] *= v_th_e # velocities
	    electrons[1,remaining_indices] += v_d_e # velocities
	    #electrons[2,remaining_indices] = np.ones(n_samples) # active slots
	    #electrons[2,remaining_indices] = np.ones(n_samples,dtype=np.float32) # relative weights
	if smooth_electron_velocities:
	    # TODO: allow quasi-random numbers here?
	    electron_v_smoothing_scale = v_th_e/10.
	    if read_electron_dist_from_file:
		electron_v_smoothing_scale = electron_v_edges[1]-electron_v_edges[0]
	    electrons[1,remaining_indices] += np.random.randn(n_samples)*electron_v_smoothing_scale
	if project_read_particles:
	    if uniform_2d_sampler.shared_seed:
		for id in range(n_engines):
		    sample = np.asarray(uniform_2d_sampler.get(n_samples)).T
		    if id==mpi_id:
			uniform_2d_sample = sample
	    elif quiet_start_and_injection:
		uniform_2d_sample = uniform_2d_sampler.get_uniform(n_samples,number_of_uniforming_bins).T
	    else:
		electron_v_smoothing_scale = electron_v_edges[1]-electron_v_edges[0]
	    electrons[1,remaining_indices] += np.random.randn(n_samples)*electron_v_smoothing_scale
	if project_read_particles:
	    if uniform_2d_sampler.shared_seed:
		for id in range(n_engines):
		    sample = np.asarray(uniform_2d_sampler.get(n_samples)).T
		    if id==mpi_id:
			uniform_2d_sample = sample
	    elif quiet_start_and_injection:
		uniform_2d_sample = uniform_2d_sampler.get_uniform(n_samples,number_of_uniforming_bins).T
	    else:
		uniform_2d_sample = uniform_2d_sampler.get(n_samples).T
	    perpendicular_velocities = norm.ppf(uniform_2d_sample[1])*v_th_e
	    velocity_magnitudes = np.sqrt(electrons[1,remaining_indices]*electrons[1,remaining_indices]
					  + perpendicular_velocities*perpendicular_velocities)
	    velocity_angles = np.arctan2(perpendicular_velocities, electrons[1,remaining_indices])
	    electrons[1,remaining_indices] = velocity_magnitudes*np.cos(velocity_angles-projection_angle)
	if create_electron_dimple:
	    # TODO: unify random number handling?
	    rejection_samples = np.random.rand(n_samples)
	    remaining_indices = remaining_indices[rejection_samples < dimple(electrons[1,remaining_indices],electrons[0,remaining_indices])]
	    n_successful_samples = n_samples - len(electrons[1,remaining_indices])
            initialize_z = False
	    if mpi_id==0 and n_electrons-n_electrons_so_far>0:
		print 'Still need', n_electrons-n_electrons_so_far, 'electrons...doing another iteration.'
	else:
	    n_successful_samples = n_samples
	n_electrons_so_far += n_successful_samples
	n_samples = n_electrons - n_electrons_so_far
    electrons[0][n_electrons:] = inactive_slot_position_flag
    # List remaining slots in reverse order to prevent memory fragmentation
    empty_electron_slots = -np.ones(electron_storage_length,dtype=np.int32)
    current_empty_electron_slot = [(electron_storage_length-n_electrons)-1]
    empty_electron_slots[0:(current_empty_electron_slot[0]+1)] = range(electron_storage_length-1,n_electrons-1,-1)
    electron_hist_n, electron_hist_n_edges = np.histogram(electrons[0][0:n_electrons], bins=electron_hist_n_edges)
    electron_hist_v, electron_hist_v_edges = np.histogram(electrons[1][0:n_electrons], bins=electron_hist_v_edges)
    comm.Allreduce(MPI.IN_PLACE, electron_hist_n, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, electron_hist_v, op=MPI.SUM)
    if (mpi_id==0):
	fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(8,2))
	for ax, data, bins in zip(axes,[electron_hist_n,electron_hist_v], [electron_hist_n_edges, electron_hist_v_edges]):
	    ax.step(bins,np.append(data,[0]), where='post')
	    filename='initialElectronDistribution.png'
	    plt.savefig(filename)


#for point, j in zip(grid, range(1+0*n_cells)):
#    ion_in_cell = np.logical_and(point<ions[0][:],ions[0][:]<=point+dz)
#    ions_in_cell = ions.T[ion_in_cell]
#    ions_in_cell = ions_in_cell.T
#    average_ion_velocity_in_cell = np.average(ions_in_cell[1])
#    electron_in_cell = np.logical_and(point<electrons[0][:],electrons[0][:]<=point+dz)
#    electrons_in_cell = electrons.T[electron_in_cell]
#    electrons_in_cell = electrons_in_cell.T
#    average_electron_velocity_in_cell = np.average(electrons_in_cell[1])
#    mpi_writeable = np.zeros(1)
#    comm.Allreduce(average_ion_velocity_in_cell, mpi_writeable, op=MPI.SUM)
#    average_ion_velocity_in_cell = mpi_writeable[0]/n_engines
#    mpi_writeable = np.zeros(1)
#    comm.Allreduce(average_electron_velocity_in_cell, mpi_writeable, op=MPI.SUM)
#    average_electron_velocity_in_cell = mpi_writeable[0]/n_engines
#    ions[1][ion_in_cell] -= average_ion_velocity_in_cell
#    electrons[1][electron_in_cell] -= average_electron_velocity_in_cell

v_drift = 0.125*v_th_i
pot_transp_elong = 2.
object_radius = 1.
t_object_center = (1.+4.*pot_transp_elong*debye_length)/v_drift
t = 0.
v_b_0 = 0. # inital box velocity
a_b_0 = 0. # initial box acceleration
if boltzmann_electrons:
    dt = 0.05*debye_length/v_th_i
else:
    dt = 0.3*debye_length/v_th_e
ion_charge_to_mass = 1
if (mpi_id==0):
    print 't_object_center, dt', t_object_center, dt


object_mask = np.zeros_like(grid) # Could make 1 shorter
object_center_mask = np.zeros_like(grid) # Could make 1 shorter
if include_object:
    circular_cross_section(grid,1.e-8,1.,1.,1.,object_center_mask)
potential = np.zeros_like(grid)
background_potential = np.linspace(2.3, 0., num=n_points, endpoint=True).astype(np.float32)
background_potential_phase2 = np.linspace(2.3, 0., num=n_points, endpoint=True).astype(np.float32)
potential_ions = potential 
if decouple_ions_from_electrons:
    potential_electrons = np.zeros_like(grid)
else:
    potential_electrons = potential
previous_potential = np.zeros_like(grid)
electric_field = np.zeros_like(grid)

if (mpi_id==0):
    object_masks = []
    potentials = []
    potentials_electrons = []
    electric_fields = []
    ion_densities = []
    electron_densities = []
    charge_derivatives = []
    times = []
    ion_distribution_functions = []
    electron_distribution_functions = []
    trapped_electron_distribution_functions = []
    

ion_density = np.zeros_like(grid)
accumulate_density(grid, object_mask, background_ion_density, largest_ion_index, ions, ion_density, \
		       empty_ion_slots, current_empty_ion_slot, periodic_particles=periodic_particles, \
		       use_pure_c_version=use_pure_c_mover)
electron_density = np.zeros_like(grid)
charge_density = np.zeros_like(grid)
previous_charge_density = np.zeros_like(grid)
charge_derivative = np.zeros_like(grid)
if not boltzmann_electrons:
    accumulate_density(grid, object_mask, background_electron_density, largest_electron_index, electrons, \
			   electron_density, empty_electron_slots, current_empty_electron_slot, \
			   periodic_particles=periodic_particles, use_pure_c_version=use_pure_c_mover)

initialize_mover(grid, object_mask, potential, dt, ion_charge_to_mass, largest_ion_index, ions, \
		     empty_ion_slots, current_empty_ion_slot, v_b=v_b_0, a_b=a_b_0, periodic_particles=periodic_particles, \
		     use_pure_c_version=use_pure_c_mover)
if not boltzmann_electrons:
    electron_charge_to_mass = -1./mass_ratio
    initialize_mover(grid, object_mask, potential_electrons, dt, electron_charge_to_mass, largest_electron_index, electrons, \
			 empty_electron_slots, current_empty_electron_slot, v_b=v_b_0, a_b=a_b_0,\
			 periodic_particles=periodic_particles, use_pure_c_version=use_pure_c_mover)
injection_numbers = np.zeros(n_engines,dtype=np.int32)


n_steps = 10000
storage_step = 10
store_all_until_step = 10
print_step = 100
#mass_correction = math.sqrt(1./1836./mass_ratio)
#debye_correction = 0.04/debye_length
#n_steps = int(150000*mass_correction*debye_correction/8.)
#storage_step = int(max(1,n_steps/100/mass_correction))
#store_all_until_step = 0
#print_step = storage_stepv
box = np.zeros([2,n_steps+1], dtype=np.float32) # initialization of box velocity and acceleration
ions_extra = np.zeros([2,n_steps+1], dtype=np.float32) # The velocities and acceleration due to background ion acceleration, zero if background acceleration is turned off.
hole_relative_positions = np.zeros(n_steps, dtype=np.float32) # initialization of electron hole relative positions
hole_velocities = np.zeros(n_steps, dtype=np.float32) # Array of electron hole relative velocities
initial_transient_steps = 10 # Hole tracking will work only when we have a fully developped hole potential, need to skip the initial transient
Wn = .002 # Cut-off frequency of Butterworth filter
N_filter = 2 # Order of Butterworth filter
#W_smoothing = 2.*debye_length/v_th_e # Rectangle smoothing window length in the control law
W_smoothing = 0.
alpha = 15.*1e-3*v_th_e**2/debye_length**2 # Control parameter on hole position 10. default
beta = 5.*v_th_e/debye_length # Control parameter on hole velocity 5. default
damping_start_step = 1 # don't make zero to avoid large initial derivative
damping_end_step = 0 # make <= damping_start_step to disable damping
B, A = signal.butter(N_filter,Wn,output='ba') # Numerator and denominator of IIR filter 
def hole_position_tracking(potential,dz,L,grid,fine_mesh,fine_dz):
    search_factor = 12
    search_center = fine_mesh.shape[0]/2
    half_search_range = search_factor*int(math.floor(L/fine_dz))
    signal = electric_field_filter(grid,np.gradient(potential,dz),L,fine_mesh)
    plausible_range_signal = signal[search_center-half_search_range:search_center+half_search_range]
    return fine_mesh[np.argmax(plausible_range_signal)+search_center-half_search_range]
if (mpi_id==0):
    print 'n_steps, n_steps*dt', n_steps, n_steps*dt
for k in range(n_steps):
    if simulate_moon:
	if k==1: # simulate moon by knocking out particles
	    # TODO: fix half-step in velocity?
	    object_mask[0.45*n_cells:0.55*n_cells] = 1.
	elif k==2:
	    object_mask = np.zeros_like(grid)
    comm.Allreduce(MPI.IN_PLACE, ion_density, op=MPI.SUM)
    if periodic_particles:
	ion_density[0] += ion_density[-1]
	ion_density[-1] = ion_density[0]
    elif k<=time_steps_immobile_ions:
	ion_density[0] *= 2. # half-cell at ends
	ion_density[-1] *= 2. # half-cell at ends
    if boltzmann_electrons:
	electron_density = np.exp(potential) # TODO: add electron temp. dep.
    else:
	comm.Allreduce(MPI.IN_PLACE, electron_density, op=MPI.SUM)
	if periodic_particles:
	    electron_density[0] += electron_density[-1]
	    electron_density[-1] = electron_density[0]
	else:
	    electron_density[0] *= 2. # half-cell at ends
	    electron_density[-1] *= 2. # half-cell at ends
    if include_object:
	dist_to_obj = circular_cross_section(grid, t, t_object_center, v_drift, object_radius, object_mask)
#    elif simulate_moon and t<=2*t_moon_center:
#	dist_to_obj = circular_cross_section(grid, t, t_moon_center, v_drift_moon, moon_radius, object_mask)
#	dist_to_obj = 100. # Don't include in potential
    else:
	dist_to_obj = 100.
    object_potential = -3.
    if prescribe_potential and t<prescribed_potential_growth_cycles*dt:
	potential = prescribed_potential(grid,t,debye_length)
    elif quasineutral:
	potential = np.log(np.maximum(ion_density,np.exp(object_potential)*np.ones_like(ion_density))) # TODO: add electron temp. dep.
	potential[0] = 0.
	potential[-1] = 0.
    elif zero_electric_field_test:
        potential = np.zeros_like(grid)
    elif solve_for_electric_field:
        np.subtract(ion_density,electron_density,out=charge_density)
        gauss_solve(grid, charge_density, debye_length, electric_field, periodic_electric_field=periodic_potential)
        np.cumsum(-electric_field*dz, out=potential)
    else:
	if (dist_to_obj>0.):
	    if include_object:
		fraction_of_obj_pot = math.exp(-dist_to_obj/debye_length)
	    else:
		fraction_of_obj_pot = 0.
	    object_potential *= math.exp(-dist_to_obj/(pot_transp_elong*debye_length))
	    np.subtract(ion_density,electron_density,out=charge_density)
	    np.subtract(charge_density,previous_charge_density,out=charge_derivative)
	    charge_derivative /= dt
	    if decouple_ions_from_electrons and t>prescribed_potential_growth_cycles*dt:
		poisson_solve(grid, object_center_mask, charge_density, \
				  debye_length, potential_electrons, \
				  object_potential=object_potential, object_transparency=(1.-fraction_of_obj_pot), \
				  boltzmann_electrons=False, periodic_potential=periodic_potential, \
				  use_pure_c_version=use_pure_c_solver)
	    np.copyto(previous_charge_density,charge_density)
	    if boltzmann_potential:
		np.copyto(charge_density,ion_density) # TODO: could just make reference
		max_potential_iter = 20
	    else:
		max_potential_iter = 1
	    if k>=damping_start_step and k<damping_end_step:
		damping_factor = dt/2.
		charge_derivative *= damping_factor
		charge_density += charge_derivative
	    else:
		damping_factor = 0.
	    potential_iter = 0
	    potential_converged = False
	    potential_convergence_threshold = 0.0001
	    while (not potential_converged and potential_iter<max_potential_iter):
		poisson_solve(grid, object_center_mask, charge_density, \
				  debye_length, potential, \
				  object_potential=object_potential, object_transparency=(1.-fraction_of_obj_pot), \
				  boltzmann_electrons=boltzmann_potential, periodic_potential=periodic_potential, \
				  use_pure_c_version=use_pure_c_solver)
		np.subtract(potential,previous_potential,out=previous_potential)
		np.absolute(previous_potential,out=previous_potential)
		max_potential_change = np.amax(previous_potential)
		np.copyto(previous_potential,potential)
		if max_potential_change<potential_convergence_threshold:
		    potential_converged = True
		potential_iter += 1
#	    if mpi_id==0:
#		print 'potential_iter, max_potential_change', potential_iter, max_potential_change
	else:
	    fraction_of_obj_pot = 1.
	    np.subtract(ion_density,electron_density,out=charge_density)
	    poisson_solve(grid, object_mask, charge_density, debye_length, potential, \
			      object_potential=object_potential, object_transparency=0., \
			      use_pure_c_version=use_pure_c_solver)
    if set_background_acceleration and k>=start_step_background_acc and k<end_step_background_acc:
        potential_ions = potential+background_potential
        ions_extra[1,k] = (background_potential[1]-background_potential[0])/dz
        ions_extra[0,k] = ions_extra[0,k-1]+ions_extra[1,k]*dt
    elif set_background_acceleration and k>=start_step_background_acc_phase2 and k<end_step_background_acc_phase2:
        potential_ions = potential+background_potential_phase2
        ions_extra[1,k] = (background_potential_phase2[1]-background_potential_phase2[0])/dz
        ions_extra[0,k] = ions_extra[0,k-1]+ions_extra[1,k]*dt
    else:
        potential_ions = potential
        ions_extra[0,k] = ions_extra[0,k-1]
    if not decouple_ions_from_electrons:
	potential_electrons = potential
    if (k%storage_step==0 or k<store_all_until_step):
	    occupied_ion_slots = (ions[0]==ions[0])
	    occupied_ion_slots[empty_ion_slots[0:current_empty_ion_slot[0]+1]] = False
	    n_bins = 100
#	    ion_hist_n_edges = np.arange(z_min,z_max+eps,(z_max-z_min)/n_bins)
	    v_bins = 100
#	    ion_hist_v_edges = np.arange(v_min_i,v_max_i+eps,(v_max_i-v_min_i)/n_bins)
	    ion_hist2d, ion_hist_n_centers, ion_hist_v_centers = \
		histogram2d_uniform_grid(ions[0][occupied_ion_slots], ions[1][occupied_ion_slots], \
				      z_min, z_max, v_min_i, v_max_i, n_bins, v_bins) # could use range=[[z_min,z_max],[v_min,v_max]]
	    ion_hist2d = np.ascontiguousarray(ion_hist2d)
	    comm.Allreduce(MPI.IN_PLACE, ion_hist2d, op=MPI.SUM)
	    if not boltzmann_electrons:
		occupied_electron_slots = (electrons[0]==electrons[0])
		occupied_electron_slots[empty_electron_slots[0:current_empty_electron_slot[0]+1]] = False
                occupied_electron_slots_inject0 = (electrons_injection_history[0]==0)
                occupied_electron_slots_inject0[empty_electron_slots[0:current_empty_electron_slot[0]+1]] = False
		n_bins = 100
#		electron_hist_n_edges = np.arange(z_min,z_max+eps,(z_max-z_min)/n_bins)
		v_bins = 100
#		electron_hist_v_edges = np.arange(v_min_e,v_max_e+eps,(v_max_e-v_min_e)/n_bins)
		electron_hist2d, electron_hist_n_centers, electron_hist_v_centers = \
		    histogram2d_uniform_grid(electrons[0][occupied_electron_slots], electrons[1][occupied_electron_slots], \
					  z_min, z_max, v_min_e, v_max_e, n_bins, v_bins) # could use range=[[z_min,z_max],[v_min,v_max]]
                electron_inject0_hist2d, electron_hist_n_centers, electron_hist_v_centers = \
		    histogram2d_uniform_grid(electrons[0][occupied_electron_slots_inject0], electrons[1][occupied_electron_slots_inject0], \
					  z_min, z_max, v_min_e, v_max_e, n_bins, v_bins) # could use range=[[z_min,z_max],[v_min,v_max]]
		electron_hist2d = np.ascontiguousarray(electron_hist2d)
                electron_inject0_hist2d = np.ascontiguousarray(electron_inject0_hist2d)
		comm.Allreduce(MPI.IN_PLACE, electron_hist2d, op=MPI.SUM)
                comm.Allreduce(MPI.IN_PLACE, electron_inject0_hist2d, op=MPI.SUM)
    if k>initial_transient_steps:
        hole_relative_positions[k] = hole_position_tracking(potential,dz,debye_length,grid,fine_mesh,fine_dz) 
    # give the relative hole position in the box by hole tracking
    if ((k%storage_step==0 or k<store_all_until_step) and mpi_id==0):
        times.append(t)
        object_masks.append(np.copy(object_mask))
        ion_densities.append(np.copy(ion_density))
	electron_densities.append(np.copy(electron_density))
	charge_derivatives.append(np.copy(charge_derivative))
        potentials.append(np.copy(potential))
	potentials_electrons.append(np.copy(potential_electrons))
        electric_fields.append(np.copy(electric_field))
        ion_distribution_functions.append(np.copy(ion_hist2d))
	if not boltzmann_electrons:
	    electron_distribution_functions.append(np.copy(electron_hist2d))
            trapped_electron_distribution_functions.append(np.copy(electron_inject0_hist2d))
    if (k%print_step==0 and mpi_id==0):
	print 'Step ', k, '  Max potentials each step:'
    hole_velocities[k] = (hole_relative_positions[k]-hole_relative_positions[k-1])/dt+box[0,k] # Apply some filtering to smooth out the hole velocity
    hole_velocities_f = signal.filtfilt(B,A,hole_velocities[0:k+1],padlen=0)
    if moving_box_simulation:
       #box[0,k+1] = np.average(box[0,max(0,k-np.floor(W_smoothing/dt)):k+1])\
                     #+alpha*dt*(hole_relative_positions[k]-0.)+beta*dt*(hole_relative_positions[k]-hole_relative_positions[k-1])/dt
       box[0,k+1] = np.average(box[0,max(0,k-np.floor(W_smoothing/dt)):k+1])\
                     +alpha*dt*(hole_relative_positions[k]-0.)+beta*dt*(hole_velocities_f[k]-box[0,k])        
# The hole velocity at time k+1/2 is an exponential smoothing result of its values at previous time steps and measured hole velocity at time k-1/2
       box[1,k] = (box[0,k+1]-box[0,k])/dt # Update box acceleration at every time step k
    if (k>time_steps_immobile_ions):
        move_particles(grid, object_mask, potential_ions, 0., ion_charge_to_mass, \
                       background_ion_density, largest_ion_index, ions, ion_density, \
		       empty_ion_slots, current_empty_ion_slot, periodic_particles=periodic_particles, a_b=box[1,k],\
		       use_pure_c_version=use_pure_c_mover)
        ion_density = np.ones_like(grid)
    else:
        move_particles(grid, object_mask, potential_ions, dt, ion_charge_to_mass, \
                       background_ion_density, largest_ion_index, ions, ion_density, \
		       empty_ion_slots, current_empty_ion_slot, periodic_particles=periodic_particles, a_b=box[1,k],\
		       use_pure_c_version=use_pure_c_mover)
    if not boltzmann_electrons:
        if (k<time_steps_immobile_electrons):
            move_particles(grid, object_mask, potential_electrons, 0., electron_charge_to_mass, \
			   background_electron_density, largest_electron_index, \
			   electrons, electron_density, empty_electron_slots, current_empty_electron_slot, a_b=box[1,k],\
			   periodic_particles=periodic_particles, use_pure_c_version=use_pure_c_mover)
        else:
            move_particles(grid, object_mask, potential_electrons, dt, electron_charge_to_mass, \
			   background_electron_density, largest_electron_index, \
			   electrons, electron_density, empty_electron_slots, current_empty_electron_slot, a_b=box[1,k],\
			   periodic_particles=periodic_particles, use_pure_c_version=use_pure_c_mover,)
    if counter_streaming_ion_beams:
        expected_ion_injection_1 = expected_particle_injection(n_ions_infinity/(z_max-z_min)/2.,v_th_i/np.sqrt(sigma),box[0,k]+ions_extra[0,k]-v_d_1,dt)
        expected_ion_injection_2 = expected_particle_injection(n_ions_infinity/(z_max-z_min)/2.,v_th_i/np.sqrt(sigma),box[0,k]+ions_extra[0,k]-v_d_2,dt)
        n_ions_inject_1 = int(expected_ion_injection_1)
        n_ions_inject_2 = int(expected_ion_injection_2)
        if (expected_ion_injection_1-n_ions_inject_1)>np.random.rand():
            n_ions_inject_1 += 1
        if (expected_ion_injection_2-n_ions_inject_2)>np.random.rand():
            n_ions_inject_2 += 1
        n_ions_inject = n_ions_inject_1+n_ions_inject_2
    else:
        expected_ion_injection = expected_particle_injection(n_ions_infinity/(z_max-z_min),v_th_i/np.sqrt(sigma),box[0,k]+ions_extra[0,k],dt)
        n_ions_inject = int(expected_ion_injection)
        if (expected_ion_injection-n_ions_inject)>np.random.rand():
            n_ions_inject += 1
    if not boltzmann_electrons:
        expected_electron_injection = expected_particle_injection(n_electrons_infinity/(z_max-z_min),v_th_e,box[0,k],dt)
	n_electrons_inject = int(expected_electron_injection)
    # If expected injection number is small, need to add randomness to get right average rate
    # TODO: unify random number usage with sampler
    #injection_numbers[:] = 0
    #injection_numbers[mpi_id] = n_ions_inject
    #comm.Allreduce(MPI.IN_PLACE, injection_numbers, op=MPI.SUM)
    injection_number = np.array(n_ions_inject,dtype=np.int32)
    comm.Allgather(injection_number, injection_numbers)
    if injection_sampler.shared_seed:
	for injection_number in injection_numbers[:mpi_id]:
	    sample = np.asarray(injection_sampler.get(int(injection_number))).T
    if n_ions_inject>0 and not periodic_particles and k<=time_steps_immobile_ions:
        if counter_streaming_ion_beams:
            inject_particles(n_ions_inject_1, grid, dt, v_th_i/np.sqrt(sigma), background_ion_density, \
			     injection_sampler, box[0,k]+ions_extra[0,k]-v_d_1, box[1,k]+ions_extra[1,k], \
                             k, ions_injection_history, ions, empty_ion_slots, \
			     current_empty_ion_slot, largest_ion_index, ion_density)
            inject_particles(n_ions_inject_2, grid, dt, v_th_i/np.sqrt(sigma), background_ion_density, \
			     injection_sampler, box[0,k]+ions_extra[0,k]-v_d_2, box[1,k]+ions_extra[1,k], \
                             k, ions_injection_history, ions, empty_ion_slots, \
			     current_empty_ion_slot, largest_ion_index, ion_density)
        else:
            inject_particles(n_ions_inject, grid, dt, v_th_i/np.sqrt(sigma), background_ion_density, \
			     injection_sampler, box[0,k]+ions_extra[0,k], box[1,k]+ions_extra[1,k], \
                             k, ions_injection_history, ions, empty_ion_slots, \
			     current_empty_ion_slot, largest_ion_index, ion_density)
    if injection_sampler.shared_seed:
	for injection_number in injection_numbers[mpi_id+1:]:
	    sample = np.asarray(injection_sampler.get(int(injection_number))).T
    if not boltzmann_electrons:
	if (expected_electron_injection-n_electrons_inject)>np.random.rand():
	    n_electrons_inject += 1
	#injection_numbers[:] = 0
        #injection_numbers[mpi_id] = n_electrons_inject
        #comm.Allreduce(MPI.IN_PLACE, injection_numbers, op=MPI.SUM)
	injection_number = np.array(n_electrons_inject,dtype=np.int32)
        comm.Allgather(injection_number, injection_numbers)
	if injection_sampler.shared_seed:
            for injection_number in injection_numbers[:mpi_id]:
		sample = np.asarray(injection_sampler.get(int(injection_number))).T
	if n_electrons_inject>0 and not periodic_particles:
	    inject_particles(n_electrons_inject, grid, dt, v_th_e, background_electron_density, \
				 injection_sampler, box[0,k], box[1,k], \
                                 k, electrons_injection_history, electrons, empty_electron_slots, \
				 current_empty_electron_slot, largest_electron_index, electron_density)
	if injection_sampler.shared_seed:
	    for injection_number in injection_numbers[mpi_id+1:]:
		sample = np.asarray(injection_sampler.get(int(injection_number))).T
    t += dt
    if(mpi_id==0):
        print  ('%.2f' % np.max(potential)) ,
        if((k+1)%10  == 0):
            print
if (mpi_id==0):
    print times[0], dt, times[len(times)-1]


if (mpi_id==0):
    times_np = np.array(times, dtype=np.float32)
    object_masks_np = np.array(object_masks, dtype=np.float32)
    potentials_np = np.array(potentials, dtype=np.float32)
    potentials_electrons_np = np.array(potentials_electrons, dtype=np.float32)
    electric_fields_np = np.array(electric_fields, dtype=np.float32)
    ion_densities_np = np.array(ion_densities, dtype=np.float32)
    electron_densities_np = np.array(electron_densities, dtype=np.float32)
    charge_derivatives_np = np.array(charge_derivatives, dtype=np.float32)
    ion_distribution_functions_np = np.array(ion_distribution_functions, dtype=np.float32) # actuall int
    electron_distribution_functions_np = np.array(electron_distribution_functions, dtype=np.float32) # actuall int
    trapped_electron_distribution_functions_np = np.array(trapped_electron_distribution_functions, dtype=np.float32)
    n_ions_total = n_ions*n_engines
    if set_background_acceleration and not counter_streaming_ion_beams: 
        filename_base = \
	'l'+('%.4f' % debye_length)+'_Vd'+('%.3f' % dimple_velocity)+'_np'+('%.1e' % n_points)+'_ni'+('%.1e' % n_ions_total)+'_dt'+('%.1e' % dt)+'_sigma'+('%.1e' % sigma)+'_nsteps'+('%.1e' %n_steps)+'_ions_acc'
    elif counter_streaming_ion_beams and not set_background_acceleration:
        filename_base = \
            'l'+('%.4f' % debye_length)+'_Vd'+('%.3f' % dimple_velocity)+'_np'+('%.1e' % n_points)+'_ni'+('%.1e' % n_ions_total)+'_dt'+('%.1e' % dt)+'_sigma'+('%.1e' % sigma)+'_nsteps'+('%.1e' %n_steps)+'_counter_beams_'+'v1_'+('%.2f' % v_d_1)+'v2_'+('%.2f' % v_d_2)
    elif counter_streaming_ion_beams and set_background_acceleration:
        filename_base = \
            'l'+('%.4f' % debye_length)+'_Vd'+('%.3f' % dimple_velocity)+'_np'+('%.1e' % n_points)+'_ni'+('%.1e' % n_ions_total)+'_dt'+('%.1e' % dt)+'_sigma'+('%.1e' % sigma)+'_nsteps'+('%.1e' %n_steps)+'_counter_beams_'+'v1_'+('%.2f' % v_d_1)+'v2_'+('%.2f' % v_d_2)+'_ions_acc'
    else:
        filename_base = \
	'l'+('%.4f' % debye_length)+'_Vd'+('%.3f' % dimple_velocity)+'_np'+('%.1e' % n_points)+'_ni'+('%.1e' % n_ions_total)+'_dt'+('%.1e' % dt)+'_sigma'+('%.1e' % sigma)+'_nsteps'+('%.1e' %n_steps)
    print filename_base
    try:
        np.savez(os.path.join(STORAGE_PATH,filename_base), grid=grid, times=times_np, object_masks=object_masks_np, potentials=potentials_np, \
                    potentials_electrons=potentials_electrons_np, electric_fields=electric_fields_np, \
		    ion_densities=ion_densities_np, electron_densities=electron_densities_np, charge_derivatives=charge_derivatives_np, \
		    ion_hist_n_centers=ion_hist_n_centers, ion_hist_v_centers=ion_hist_v_centers, \
		    ion_distribution_functions=ion_distribution_functions_np, \
		    electron_hist_n_centers=electron_hist_n_centers, electron_hist_v_centers=electron_hist_v_centers, \
		    electron_distribution_functions=electron_distribution_functions_np, \
                    trapped_electron_distribution_functions=trapped_electron_distribution_functions_np, \
		    background_ion_density=background_ion_density, background_electron_density=background_electron_density, \
                    box_velocities=box[0], hole_relative_positions=hole_relative_positions, hole_velocities=hole_velocities, ions_vel=ions_extra)
    except:
        np.savez(filename_base, grid=grid, times=times_np, object_masks=object_masks_np, potentials=potentials_np, \
                    potentials_electrons=potentials_electrons_np, electric_fields=electric_fields_np, \
		    ion_densities=ion_densities_np, electron_densities=electron_densities_np, charge_derivatives=charge_derivatives_np, \
		    ion_hist_n_centers=ion_hist_n_centers, ion_hist_v_centers=ion_hist_v_centers, \
		    ion_distribution_functions=ion_distribution_functions_np, \
		    electron_hist_n_centers=electron_hist_n_centers, electron_hist_v_centers=electron_hist_v_centers, \
		    electron_distribution_functions=electron_distribution_functions_np, \
                    trapped_electron_distribution_functions=trapped_electron_distribution_functions_np, \
		    background_ion_density=background_ion_density, background_electron_density=background_electron_density, \
                    box_velocities=box[0], hole_relative_positions=hole_relative_positions, hole_velocities=hole_velocities, ions_vel=ions_extra)

