# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from IPython.parallel import Client

# <codecell>

rc = Client()

# <codecell>

%%px
import IPython.core.display as IPdisp

# <codecell>

%%px
from IPython.nbformat import current
def execute_notebook(nbfile):
    with io.open(nbfile) as f:
        nb = current.read(f, 'json')
    ip = get_ipython()
    for cell in nb.worksheets[0].cells:
        if cell.cell_type != 'code':
            continue
        ip.run_cell(cell.input)

# <codecell>

#####ScriptStart#####

# <codecell>

%%px
from mpi4py import MPI
import io
import math
import numpy
import matplotlib
matplotlib.use('Agg') # non-GUI backend
import matplotlib.pyplot
import ghalton
from scipy.stats import norm
from infMagSim_cython import *

# <codecell>

%%px
comm = MPI.COMM_WORLD
mpi_id = comm.Get_rank()
n_engines = comm.size

# <codecell>

%%px
periodic_particles = False # only use with prescribe_potential for now
prescribe_potential = False or periodic_particles
def prescribed_potential(grid, t):
    z_min = grid[0]
    z_max = grid[-1]
    return min(1.,t/(z_max-z_min))*np.sin(2.*np.pi*grid/(z_max-z_min))
boltzmann_electrons = False
quasineutral = False
if quasineutral:
    boltzmann_electrons = True
use_quasirandom_numbers = False
include_object = True

# <codecell>

%%px
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

# <codecell>

%%px
z_min = -50.
z_max = 50.
n_cells = 2000
n_points = n_cells+1
dz = (z_max-z_min)/(n_points-1)
eps = 1e-4
grid = numpy.arange(z_min,z_max+eps,dz,dtype=numpy.float32)
if (mpi_id==0):
    print dz

# <codecell>

%%px
seed = 384+mpi_id*mpi_id*mpi_id
class uniform_2d_sampler_class:
    def __init__(self,seed,low_discrepancy=False,shared_seed=True,rand_dim=1):
	self.low_discrepancy = low_discrepancy
	self.shared_seed = shared_seed
	self.rand_dim = rand_dim
	if low_discrepancy:
	    self.low_discrepancy = True
	    #self.sequencer = ghalton.GeneralizedHalton(rand_dim,seed) # seed doesn't appear to do anything
	    self.sequencer = sobol_sequencer(rand_dim=rand_dim)
	else:
	    self.low_discrepancy = False
	    numpy.random.seed(seed)
    def get(self,n):
	if self.low_discrepancy:
	    return numpy.asarray(self.sequencer.get(n))
	else:
	    return numpy.random.rand(n,self.rand_dim)
if use_quasirandom_numbers:
    uniform_2d_sampler = uniform_2d_sampler_class(seed,low_discrepancy=True,rand_dim=2)
else:
    uniform_2d_sampler = uniform_2d_sampler_class(seed,shared_seed=False,rand_dim=2)

# <codecell>

%%px
v_th_i = 1
v_d_i = 0
n_ions = 100000
extra_storage_factor = 1.2
background_ion_density = n_ions*n_engines*dz/(z_max-z_min)
n_bins = n_points
v_max_i = 4.*v_th_i
v_min_i = -v_max_i
inactive_slot_position_flag = 2*z_max

# <codecell>

%%px
largest_ion_index = [n_ions-1]
ion_storage_length = int(extra_storage_factor*n_ions)
ions = numpy.zeros([2,ion_storage_length],dtype=numpy.float32)
#ions[0][0:n_ions] = numpy.random.rand(n_ions)*(z_max-z_min) + z_min # positions
#ions[1][0:n_ions] = numpy.random.randn(n_ions)*v_th_i + v_d_i # velocities
if uniform_2d_sampler.shared_seed:
    for id in range(n_engines):
	sample = numpy.asarray(uniform_2d_sampler.get(n_ions)).T
	if id==mpi_id:
	    uniform_2d_sample = sample
else:
    uniform_2d_sample = uniform_2d_sampler.get(n_ions).T
ions[0][0:n_ions] = uniform_2d_sample[0]*(z_max-z_min) + z_min # positions
ions[1][0:n_ions] = norm.ppf(uniform_2d_sample[1])*v_th_i + v_d_i # velocities
#ions[2][0:n_ions] = np.ones(n_ions) # active slots
ions[0][n_ions:] = inactive_slot_position_flag
#ions[2][0:n_ions] = numpy.ones(n_ions,dtype=numpy.float32) # relative weights
# List remaining slots in reverse order to prevent memory fragmentation
empty_ion_slots = -numpy.ones(ion_storage_length,dtype=numpy.int)
current_empty_ion_slot = [(ion_storage_length-n_ions)-1]
empty_ion_slots[0:(current_empty_ion_slot[0]+1)] = range(ion_storage_length-1,n_ions-1,-1)
ion_hist_n_edges = np.arange(z_min,z_max+eps,(z_max-z_min)/n_bins)
ion_hist_v_edges = np.arange(v_min_i,v_max_i+eps,(v_max_i-v_min_i)/n_bins)
ion_hist_n, ion_hist_n_edges = numpy.histogram(ions[0][0:n_ions], bins=ion_hist_n_edges)
ion_hist_v, ion_hist_v_edges = numpy.histogram(ions[1][0:n_ions], bins=ion_hist_v_edges)
comm.Allreduce(MPI.IN_PLACE, ion_hist_n, op=MPI.SUM)
comm.Allreduce(MPI.IN_PLACE, ion_hist_v, op=MPI.SUM)
if (mpi_id==0):
    fig, axes = matplotlib.pyplot.subplots(nrows=1,ncols=2,figsize=(8,2))
    for ax, data, bins in zip(axes,[ion_hist_n,ion_hist_v], [ion_hist_n_edges, ion_hist_v_edges]):
	ax.step(bins,numpy.append(data,[0]), where='post')
    filename='initialIonDistribution.png'
    matplotlib.pyplot.savefig(filename)

# <codecell>

%%px
#if not boltzmann_electrons:
mass_ratio = 1./1836.
n_electrons = n_ions
v_th_e = 1./math.sqrt(mass_ratio)
v_d_e = 0.
background_electron_density = n_electrons*n_engines*dz/(z_max-z_min)
n_bins = n_points;
v_max_e = 4.*v_th_e
v_min_e = -v_max_e

# <codecell>

%%px
electron_hist_n_edges = np.arange(z_min,z_max+eps,(z_max-z_min)/n_bins)
electron_hist_v_edges = np.arange(v_min_e,v_max_e+eps,(v_max_e-v_min_e)/n_bins)
if not boltzmann_electrons:
    largest_electron_index = [n_electrons-1]
    electron_storage_length = int(n_electrons*extra_storage_factor)
    electrons = numpy.zeros([2,electron_storage_length],dtype=numpy.float32)
    #electrons[0][0:n_electrons] = numpy.random.rand(n_electrons)*(z_max-z_min) + z_min # positions
    #electrons[1][0:n_electrons] = numpy.random.randn(n_electrons)*v_th_e + v_d_e # velocities
    if uniform_2d_sampler.shared_seed:
	for id in range(n_engines):
	    sample = numpy.asarray(uniform_2d_sampler.get(n_electrons)).T
	    if id==mpi_id:
		uniform_2d_sample = sample
    else:
	uniform_2d_sample = uniform_2d_sampler.get(n_electrons).T
    electrons[0][0:n_electrons] = uniform_2d_sample[0]*(z_max-z_min) + z_min # positions
    electrons[1][0:n_electrons] = norm.ppf(uniform_2d_sample[1])*v_th_e + v_d_e # velocities
    #electrons[2][0:n_electrons] = np.ones(n_electrons) # active slots
    electrons[0][n_electrons:] = inactive_slot_position_flag
    #electrons[2][0:n_electrons] = numpy.ones(n_ions,dtype=numpy.float32) # relative weights
    # List remaining slots in reverse order to prevent memory fragmentation
    empty_electron_slots = -numpy.ones(electron_storage_length,dtype=numpy.int)
    current_empty_electron_slot = [(electron_storage_length-n_electrons)-1]
    empty_electron_slots[0:(current_empty_electron_slot[0]+1)] = range(electron_storage_length-1,n_electrons-1,-1)
    electron_hist_n, electron_hist_n_edges = numpy.histogram(electrons[0][0:n_electrons], bins=electron_hist_n_edges)
    electron_hist_v, electron_hist_v_edges = numpy.histogram(electrons[1][0:n_electrons], bins=electron_hist_v_edges)
    comm.Allreduce(MPI.IN_PLACE, electron_hist_n, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, electron_hist_v, op=MPI.SUM)
    if (mpi_id==0):
	fig, axes = matplotlib.pyplot.subplots(nrows=1,ncols=2,figsize=(8,2))
	for ax, data, bins in zip(axes,[electron_hist_n,electron_hist_v], [electron_hist_n_edges, electron_hist_v_edges]):
	    ax.step(bins,numpy.append(data,[0]), where='post')
	    filename='initialElectronDistribution.png'
	    matplotlib.pyplot.savefig(filename)

# <codecell>

#%%px
#for point, j in zip(grid, range(1+0*n_cells)):
#    ion_in_cell = numpy.logical_and(point<ions[0][:],ions[0][:]<=point+dz)
#    ions_in_cell = ions.T[ion_in_cell]
#    ions_in_cell = ions_in_cell.T
#    average_ion_velocity_in_cell = numpy.average(ions_in_cell[1])
#    electron_in_cell = numpy.logical_and(point<electrons[0][:],electrons[0][:]<=point+dz)
#    electrons_in_cell = electrons.T[electron_in_cell]
#    electrons_in_cell = electrons_in_cell.T
#    average_electron_velocity_in_cell = numpy.average(electrons_in_cell[1])
#    mpi_writeable = numpy.zeros(1)
#    comm.Allreduce(average_ion_velocity_in_cell, mpi_writeable, op=MPI.SUM)
#    average_ion_velocity_in_cell = mpi_writeable[0]/n_engines
#    mpi_writeable = numpy.zeros(1)
#    comm.Allreduce(average_electron_velocity_in_cell, mpi_writeable, op=MPI.SUM)
#    average_electron_velocity_in_cell = mpi_writeable[0]/n_engines
#    ions[1][ion_in_cell] -= average_ion_velocity_in_cell
#    electrons[1][electron_in_cell] -= average_electron_velocity_in_cell

# <codecell>

%%px
v_drift = 0.125*v_th_i
if quasineutral:
    debye_length = dz
else:
    debye_length = 0.125
pot_transp_elong = 2.
object_radius = 1.
t_object_center = (1.+4.*pot_transp_elong*debye_length)/v_drift
t = 0.
if boltzmann_electrons:
    dt = 0.05*debye_length/v_th_i
else:
    dt = 0.1*debye_length/v_th_e
ion_charge_to_mass = 1
if (mpi_id==0):
    print t_object_center, dt

# <codecell>

%%px
object_mask = numpy.zeros_like(grid) # Could make 1 shorter
object_center_mask = numpy.zeros_like(grid) # Could make 1 shorter
if include_object:
    circular_cross_section(grid,1.e-8,1.,1.,1.,object_center_mask)
potential = numpy.zeros_like(grid)
previous_potential = numpy.zeros_like(grid)

if (mpi_id==0):
    object_masks = []
    potentials = []
    ion_densities = []
    electron_densities = []
    charge_derivatives = []
    times = []
    ion_distribution_functions = []
    electron_distribution_functions = []

# <codecell>

%%px
ion_density = numpy.zeros_like(grid)
accumulate_density(grid, object_mask, background_ion_density, largest_ion_index, ions, ion_density)
electron_density = numpy.zeros_like(grid)
charge_density = numpy.zeros_like(grid)
previous_charge_density = numpy.zeros_like(grid)
charge_derivative = numpy.zeros_like(grid)
if not boltzmann_electrons:
    accumulate_density(grid, object_mask, background_electron_density, largest_electron_index, electrons, electron_density)

initialize_mover(grid, object_mask, potential, dt, ion_charge_to_mass, largest_ion_index, ions, \
		     periodic_particles=periodic_particles)
if not boltzmann_electrons:
    electron_charge_to_mass = -1./mass_ratio
    initialize_mover(grid, object_mask, potential, dt, electron_charge_to_mass, largest_electron_index, electrons, \
			 periodic_particles=periodic_particles)

# <codecell>

%%px
n_steps = 1000
storage_step = 1
store_all_until_step = 100
print_step = 100
damping_start_step = 1 # don't make zero to avoid large initial derivative
damping_end_step = 0 # make <= damping_start_step to disable damping
for k in range(n_steps):
#    if k==1: # simulate moon by knocking out particles
#        object_mask[0.45*n_cells:0.55*n_cells] = 1.
#    else:
#        object_mask = numpy.zeros_like(grid)
    comm.Allreduce(MPI.IN_PLACE, ion_density, op=MPI.SUM)
    if periodic_particles:
	ion_density[0] += ion_density[-1]
	ion_density[-1] = ion_density[0]
    else:
	ion_density[0] *= 2. # half-cell at ends
	ion_density[-1] *= 2. # half-cell at ends
    if boltzmann_electrons:
	electron_density = numpy.exp(potential) # TODO: add electron temp. dep.
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
    else:
	dist_to_obj = 100.
    object_potential = -3.
    if prescribe_potential:
	potential = prescribed_potential(grid,t)
    elif quasineutral:
	potential = numpy.log(numpy.maximum(ion_density,numpy.exp(object_potential)*numpy.ones_like(ion_density))) # TODO: add electron temp. dep.
	potential[0] = 0.
	potential[-1] = 0.
    else:
	if (dist_to_obj>0.):
	    if include_object:
		fraction_of_obj_pot = math.exp(-dist_to_obj/debye_length)
	    else:
		fraction_of_obj_pot = 0.
	    object_potential *= math.exp(-dist_to_obj/(pot_transp_elong*debye_length))
	    if k>=damping_start_step and k<damping_end_step:
		damping_factor = dt/2.
	    else:
		damping_factor = 0.
	    charge_density = ion_density-electron_density
	    charge_derivative = (charge_density-previous_charge_density)/dt
	    previous_charge_density = charge_density
	    if boltzmann_electrons:
		charge_density = ion_density
		max_potential_iter = 20
	    else:
		max_potential_iter = 1
	    potential_iter = 0
	    potential_converged = False
	    potential_convergence_threshold = 0.0001
	    while (not potential_converged and potential_iter<max_potential_iter):
		poisson_solve(grid, object_center_mask, charge_density+damping_factor*charge_derivative, \
				  debye_length, potential, \
				  object_potential=object_potential, object_transparency=(1.-fraction_of_obj_pot), \
				  boltzmann_electrons=boltzmann_electrons)
		max_potential_change = np.amax(np.fabs(potential-previous_potential))
		previous_potential[:] = potential[:]
		if max_potential_change<potential_convergence_threshold:
		    potential_converged = True
		potential_iter += 1
#	    if mpi_id==0:
#		print potential_iter, max_potential_change
	else:
	    fraction_of_obj_pot = 1.
	    poisson_solve(grid, object_mask, ion_density-electron_density, debye_length, potential, \
			      object_potential=object_potential, object_transparency=0.)
    if (k%storage_step==0 or k<store_all_until_step):
	    occupied_ion_slots = (ions[0]==ions[0])
	    occupied_ion_slots[empty_ion_slots[0:current_empty_ion_slot[0]+1]] = False
	    n_bins = 100
	    ion_hist_n_edges = np.arange(z_min,z_max+eps,(z_max-z_min)/n_bins)
	    ion_hist_v_edges = np.arange(v_min_i,v_max_i+eps,(v_max_i-v_min_i)/n_bins)
	    ion_hist2d, ion_hist_n_edges, ion_hist_v_edges = \
		numpy.histogram2d(ions[0][occupied_ion_slots], ions[1][occupied_ion_slots], \
				      bins=[ion_hist_n_edges,ion_hist_v_edges]) # could use range=[[z_min,z_max],[v_min,v_max]]
	    ion_hist2d = np.ascontiguousarray(ion_hist2d)
	    comm.Allreduce(MPI.IN_PLACE, ion_hist2d, op=MPI.SUM)
	    if not boltzmann_electrons:
		occupied_electron_slots = (electrons[0]==electrons[0])
		occupied_electron_slots[empty_electron_slots[0:current_empty_electron_slot[0]+1]] = False
		n_bins = 100
		electron_hist_n_edges = np.arange(z_min,z_max+eps,(z_max-z_min)/n_bins)
		electron_hist_v_edges = np.arange(v_min_e,v_max_e+eps,(v_max_e-v_min_e)/n_bins)
		electron_hist2d, electron_hist_n_edges, electron_hist_v_edges = \
		    numpy.histogram2d(electrons[0][occupied_electron_slots], electrons[1][occupied_electron_slots], \
					  bins=[electron_hist_n_edges,electron_hist_v_edges]) # could use range=[[z_min,z_max],[v_min,v_max]]
		electron_hist2d = np.ascontiguousarray(electron_hist2d)
		comm.Allreduce(MPI.IN_PLACE, electron_hist2d, op=MPI.SUM)
    if ((k%storage_step==0 or k<store_all_until_step) and mpi_id==0):
        times.append(t)
        copy = numpy.empty_like(object_mask)
        copy[:] = object_mask
        object_masks.append(copy)
        copy = numpy.empty_like(ion_density)
        copy[:] = ion_density
        ion_densities.append(copy)
	copy = numpy.empty_like(electron_density)
	copy[:] = electron_density
	electron_densities.append(copy)
	copy = numpy.empty_like(charge_derivative)
	copy[:] = charge_derivative
	charge_derivatives.append(copy)
        copy = numpy.empty_like(potential)
        copy[:] = potential
        potentials.append(copy)
        copy = numpy.empty_like(ion_hist2d)
        copy[:] = ion_hist2d
        ion_distribution_functions.append(copy)
	if not boltzmann_electrons:
	    copy = numpy.empty_like(electron_hist2d)
	    copy[:] = electron_hist2d
	    electron_distribution_functions.append(copy)
    if (k%print_step==0 and mpi_id==0):
	print k
    move_particles(grid, object_mask, potential, dt, ion_charge_to_mass, \
			   background_ion_density, largest_ion_index, ions, ion_density, \
			   empty_ion_slots, current_empty_ion_slot, periodic_particles=periodic_particles)
    if not boltzmann_electrons:
	move_particles(grid, object_mask, potential, dt, electron_charge_to_mass, \
			   background_electron_density, largest_electron_index, \
			   electrons, electron_density, empty_electron_slots, current_empty_electron_slot, \
			   periodic_particles=periodic_particles)
    expected_ion_injection = 2*dt*v_th_i/math.sqrt(2*math.pi)*n_ions/(z_max-z_min)
    n_ions_inject = int(expected_ion_injection)
    if not boltzmann_electrons:
	expected_electron_injection = 2*dt*v_th_e/math.sqrt(2*math.pi)*n_electrons/(z_max-z_min)
	n_electrons_inject = int(expected_electron_injection)
    # If expected injection number is small, need to add randomness to get right average rate
    if (expected_ion_injection-n_ions_inject)>numpy.random.rand():
	n_ions_inject += 1
    injection_numbers = numpy.zeros(n_engines,dtype=np.int32)
    injection_numbers[mpi_id] = n_ions_inject
    comm.Allreduce(MPI.IN_PLACE, injection_numbers, op=MPI.SUM)
    if uniform_2d_sampler.shared_seed:
	for injection_number in injection_numbers[:mpi_id]:
	    sample = numpy.asarray(uniform_2d_sampler.get(int(injection_number))).T
    if n_ions_inject>0 and not periodic_particles:
	inject_particles(n_ions_inject, grid, dt, v_th_i, background_ion_density, \
			     uniform_2d_sampler, ions, empty_ion_slots, \
			     current_empty_ion_slot, largest_ion_index, ion_density)
    if uniform_2d_sampler.shared_seed:
	for injection_number in injection_numbers[mpi_id+1:]:
	    sample = numpy.asarray(uniform_2d_sampler.get(int(injection_number))).T
    if not boltzmann_electrons:
	if (expected_electron_injection-n_electrons_inject)>numpy.random.rand():
	    n_electrons_inject += 1
	injection_numbers = numpy.zeros(n_engines,dtype=np.int32)
	injection_numbers[mpi_id] = n_electrons_inject
	comm.Allreduce(MPI.IN_PLACE, injection_numbers, op=MPI.SUM)
	if uniform_2d_sampler.shared_seed:
	    for injection_number in injection_numbers[:mpi_id]:
		sample = numpy.asarray(uniform_2d_sampler.get(int(injection_number))).T
	if n_electrons_inject>0 and not periodic_particles:
	    inject_particles(n_electrons_inject, grid, dt, v_th_e, background_electron_density, \
				 uniform_2d_sampler, electrons, empty_electron_slots, \
				 current_empty_electron_slot, largest_electron_index, electron_density)
	if uniform_2d_sampler.shared_seed:
	    for injection_number in injection_numbers[mpi_id+1:]:
		sample = numpy.asarray(uniform_2d_sampler.get(int(injection_number))).T
    t += dt
if (mpi_id==0):
    print times[0], dt, times[len(times)-1]

# <codecell>

%%px
if (mpi_id==0):
    times_np = numpy.array(times, dtype=numpy.float32)
    object_masks_np = numpy.array(object_masks, dtype=numpy.float32)
    potentials_np = numpy.array(potentials, dtype=numpy.float32)
    ion_densities_np = numpy.array(ion_densities, dtype=numpy.float32)
    electron_densities_np = numpy.array(electron_densities, dtype=numpy.float32)
    charge_derivatives_np = numpy.array(charge_derivatives, dtype=numpy.float32)
    ion_distribution_functions_np = numpy.array(ion_distribution_functions, dtype=numpy.float32) # actuall int
    electron_distribution_functions_np = numpy.array(electron_distribution_functions, dtype=numpy.float32) # actuall int
    filename_base = \
	'l'+('%.4f' % debye_length)+'_d'+('%.3f' % v_drift)+'_np'+('%.1e' % n_points)+'_ni'+('%.1e' % n_ions)+'_dt'+('%.1e' % dt)
    print filename_base
    numpy.savez(filename_base, grid=grid, times=times_np, object_masks=object_masks_np, potentials=potentials_np, \
		    ion_densities=ion_densities_np, electron_densities=electron_densities_np, charge_derivatives=charge_derivatives_np, \
		    ion_hist_n_edges=ion_hist_n_edges, ion_hist_v_edges=ion_hist_v_edges, \
		    ion_distribution_functions=ion_distribution_functions_np, \
		    electron_hist_n_edges=electron_hist_n_edges, electron_hist_v_edges=electron_hist_v_edges, \
		    electron_distribution_functions=electron_distribution_functions_np, \
		    background_ion_density=background_ion_density, background_electron_density=background_electron_density)

# <codecell>

#####ScriptEnd#####

# <codecell>

from IPython.parallel import Client

# <codecell>

rc = Client()
dview = rc[:]
dview.block = True
n_engines = len(rc.ids)
res = dview.push(dict(n_engines=n_engines))
print n_engines

# <codecell>

mpi_ids = numpy.array(dview.pull('mpi_id'))
master_rc_id = numpy.arange(0,len(mpi_ids))[mpi_ids==0][0]

# <codecell>

filename = dview.pull('filename_base', targets=master_rc_id) + '.npz'
data_file = numpy.load(filename)
print data_file.files
grid = data_file['grid']
times = data_file['times']
ion_densities = data_file['ion_densities']
electron_densities = data_file['electron_densities']
potentials = data_file['potentials']
object_masks = data_file['object_masks']

# <codecell>

k = len(times)-1
print times[k]
fig, axes = matplotlib.pyplot.subplots(nrows=3,ncols=2,figsize=(8,6))
for ax, data in zip(axes.flatten(),[potentials[k], ion_densities[k]-electron_densities[k], \
                              object_masks[k], object_masks[k], ion_densities[k], electron_densities[k]]):
    ax.plot(grid,data)
filename='figures/data.png'
matplotlib.pyplot.savefig(filename)
IPdisp.Image(filename=filename)

# <codecell>

# 8 engines: 14.4s
# 16 engines: 18.6s
# 32 engines: 29.9s
# 1 eng, 10x part: 1m58
# 2 eng, 10x part: 1m54
# 4 eng, 10x part: 2m31
# 8 eng, 10x part: 2m8; 2m6
# 15 eng, 5x part: 1m10
# 16 eng, 5x part: 1m17; 1m17
# 16 eng, 5x part, 10x grid: 1m18
# 16 eng, 5x steps: 1m31
# 16 eng, 10x grid, 5x steps: 1m37
# 16 eng, 5x part, 10x grid, 100x steps: 2h7

# <codecell>

from guppy import hpy
h = hpy()
print h.heap()

# <codecell>

import gc

# <codecell>

gc.collect()

# <codecell>

fig, axes = matplotlib.pyplot.subplots(nrows=1,ncols=2,figsize=(8,2))
for ax, data in zip(axes,electrons):
    n_bins = n_points;
    occupied_slots = (data==data)
    occupied_slots[empty_electron_slots[0:current_empty_electron_slot[0]+1]] = False
    ax.hist(data[occupied_slots],bins=n_bins, histtype='step')
filename='data.png'
matplotlib.pyplot.savefig(filename)
IPdisp.Image(filename=filename)

# <codecell>

fig, axes = matplotlib.pyplot.subplots(nrows=1,ncols=2,figsize=(8,2))
for ax, data in zip(axes,ions):
    n_bins = n_points;
    occupied_slots = (data==data)
    occupied_slots[empty_ion_slots[0:current_empty_ion_slot[0]+1]] = False
    ax.hist(data[occupied_slots],bins=n_bins, histtype='step')
filename='data.png'
matplotlib.pyplot.savefig(filename)
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

#dir(MPI.COMM_WORLD)
help(MPI.COMM_WORLD.Allreduce)

# <codecell>


