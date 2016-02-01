# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from math import *
import numpy as np
import matplotlib
import bisect
import sys
import glob
matplotlib.use('Qt4Agg') # GUI backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
import IPython.core.display as IPdisp

# <codecell>

folder = "./"
file_names = glob.glob(folder + "l*.npz")
file_names.sort()
i=0
for file_name in file_names:
    print i, file_name
    i += 1

# <codecell>

file_index = -1

#plot_dist_func = False
plot_dist_func = True
file_name = file_names[file_index]
data_file = np.load(file_name)
print data_file.files
grid = data_file['grid']
times = data_file['times']
ion_densities = data_file['ion_densities']
electron_densities = data_file['electron_densities']
#charge_derivatives = data_file['charge_derivatives']
potentials = data_file['potentials']
#potentials_electrons = data_file['potentials_electrons']
if file_index>82:
    electric_fields = data_file['electric_fields']
else:
    electric_fields = potentials
object_masks = data_file['object_masks']
if plot_dist_func:
    ion_hist_n_edges = data_file['ion_hist_n_edges']
    ion_hist_v_edges = data_file['ion_hist_v_edges']
    ion_distribution_functions = data_file['ion_distribution_functions']
    electron_hist_n_edges = data_file['electron_hist_n_edges']
    electron_hist_v_edges = data_file['electron_hist_v_edges']
    electron_distribution_functions = data_file['electron_distribution_functions']
    background_ion_density = data_file['background_ion_density']
    background_electron_density = data_file['background_electron_density']

# <codecell>

def slider_changed(new_value, data_arrays, plots, fig, times=[], texts=[]):
    k = int(new_value)
    for plot, data in zip(plots,data_arrays):
	plot.set_ydata(data[k])
    for txt in texts:
        txt.set_text("t = {0}".format(times[k]))
    fig.canvas.draw()

def arrow_pressed(event, data_arrays, plots, fig, slider, times=[], texts=[]):
    value = slider.val
    if (event.key=='left' or event.key=='4'):
	value -= 1
    elif (event.key=='right' or event.key=='6'):
	value += 1
    elif (event.key=='down' or event.key=='5'):
	value -= 1
    elif (event.key=='up' or event.key=='8'):
	value += 1
    slider.set_val(value)
    slider_changed(value, data_arrays, plots, fig, times, texts)

k = len(times)-1
print times[k]
fig = plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(4, 3, height_ratios=[16,16,1,1])
axes = []
plots = []
if plot_dist_func:
    if len(electron_distribution_functions)>0:
	x_axes = [grid, ion_hist_v_edges[:-1], electron_hist_v_edges[:-1], grid, grid, grid, grid]
    else:
	x_axes = [grid, ion_hist_v_edges[:-1], ion_hist_v_edges[:-1], grid, grid, grid, grid]
else:
    x_axes = [grid, grid, grid, grid, grid, grid, grid]
if plot_dist_func:
    if len(electron_distribution_functions)>0:
	data_arrays = [potentials, \
			   np.squeeze(ion_distribution_functions[:,(ion_distribution_functions.shape[1]-1)*4./8.,:]), \
			   np.squeeze(electron_distribution_functions[:,(electron_distribution_functions.shape[1]-1)*4./8.,:]), \
#			   electric_fields, ion_densities, electron_densities]
			   ion_densities-electron_densities, ion_densities, electron_densities]
	#offset = 47
	#offset = 160
	#offset = 280
	#data_arrays = [potentials, \
	#		   np.squeeze(ion_distribution_functions[:,(len(grid)/2+offset)/10,:]), \
	#		   (np.squeeze(electron_distribution_functions[:,(len(grid)/2+offset)/10,:])).copy(), \
	#		   ion_densities-electron_densities, ion_densities, electron_densities]
	#electron_distribution_functions[:,(len(grid)/2+offset)/10,:] = 0
    else:
	data_arrays = [potentials, \
			   np.squeeze(ion_distribution_functions[:,(ion_distribution_functions.shape[1]-1)*5./8.,:]), \
			   np.squeeze(ion_distribution_functions[:,(ion_distribution_functions.shape[1]-1)*5./8.,:]), \
			   electric_fields, ion_densities, electron_densities]
#			   ion_densities-electron_densities, ion_densities, electron_densities]
else:
    data_arrays = [potentials, \
		       object_masks, object_masks, \
		       electric_fields, ion_densities, electron_densities]
#		       ion_densities-electron_densities, ion_densities, electron_densities]
#data_arrays = [potentials, ion_densities-electron_densities, object_masks, charge_derivatives, ion_densities, electron_densities]
ranges = []
for data in data_arrays:
    this_range = [np.nanmin(data[np.fabs(data)<np.inf]),np.nanmax(data[np.fabs(data)<np.inf])]
    #print this_range
    ranges.append(this_range)
for spec, data, this_range, x in zip(gs, data_arrays, ranges, x_axes):
    ax = plt.subplot(spec)
    ax.set_ylim(this_range)
    #ax.set_ylim([-5.1,5.1])
    axes.append(ax)
    if data.shape[1]==len(x):
	plot, = ax.plot(x,data[k])
    else:
	plot, = ax.plot(data[k])
    plots.append(plot)

texts = []
ax = plt.subplot(gs[2,:])
ax.axis('off')
txt = ax.text(0,0,"t = {0}".format(times[k]))
texts.append(txt)

ax = plt.subplot(gs[3,:])
slider = Slider(ax, 'k', 0, k, valinit=len(times)-1, color='#AAAAAA')
changed_function = partial(slider_changed, data_arrays=data_arrays, plots=plots, fig=fig, times=times, texts=texts)
slider.on_changed(changed_function)

arrow_pressed_function = partial(arrow_pressed, data_arrays=data_arrays, plots=plots, fig=fig, slider=slider, times=times, texts=texts)
fig.canvas.mpl_connect('key_press_event',arrow_pressed_function)

#plt.tight_layout()
plt.show()

# <codecell>

def slider_changed(new_value, data_arrays, plots, fig, times, texts):
    k = int(new_value)
    for plot, data in zip(plots,data_arrays):
	plot.set_data(data[k].T)
    for txt in texts:
        txt.set_text("t = {0}".format(times[k]))
    fig.canvas.draw()

#k=len(times)-1
k=0
fig = plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(4, 1, height_ratios=[16,16,1,1])
if len(electron_distribution_functions)>0:
    v_th_e = electron_hist_v_edges[-1]/4. # TODO: do this in less hacky way
    electron_bin_velocities = (electron_hist_v_edges[:-1]+electron_hist_v_edges[1:])/2.
    electron_maxwellian = np.ones_like(electron_distribution_functions)
    electron_maxwellian *= np.exp(-electron_bin_velocities*electron_bin_velocities/2./v_th_e/v_th_e)
    electron_maxwellian /= v_th_e*np.sqrt(2.*np.pi)
    electron_maxwellian *= np.sum(electron_distribution_functions,axis=2)[:,:,None]
    electron_maxwellian *= (electron_hist_v_edges[-1]-electron_hist_v_edges[0])/electron_hist_v_edges.shape[0]
    #data_arrays = [ion_distribution_functions, electron_distribution_functions]
    data_arrays = [ion_distribution_functions, electron_distribution_functions-electron_maxwellian]
else:
    data_arrays = [ion_distribution_functions, ion_distribution_functions]
plots = []
ax = plt.subplot(gs[0,0])
extent = [ion_hist_n_edges[0], ion_hist_n_edges[-1], ion_hist_v_edges[0], ion_hist_v_edges[-1] ]
im = ax.imshow(data_arrays[0][k].T,origin='lower',extent=extent,cmap='copper')
plots.append(im)
ax.set_aspect('auto')
ax.set_xlabel('$y\ [R_\mathrm{M}]$', fontsize=16)
ax.set_ylabel(r'$v_y\ [\frac{R_\mathrm{M}}{v_\mathrm{T\,i}}]$', fontsize=16)
cbar = fig.colorbar(im)
cbar.set_label(r'$\frac{f_\mathrm{e}}{f_\mathrm{e\,\infty}(0)}$', fontsize=16)
ax = plt.subplot(gs[1,0])
extent = [electron_hist_n_edges[0], electron_hist_n_edges[-1], electron_hist_v_edges[0], electron_hist_v_edges[-1] ]
im = ax.imshow(data_arrays[1][k].T,origin='lower',extent=extent,cmap='pink',vmin=np.amin(data_arrays[1])/8.,vmax=np.amax(data_arrays[1])/8.)
plots.append(im)
ax.set_aspect('auto')
fig.colorbar(im)

texts = []
ax = plt.subplot(gs[2,:])
ax.axis('off')
txt = ax.text(0,0,"t = {0}".format(times[k]))
texts.append(txt)

ax = plt.subplot(gs[3,:])
slider = Slider(ax, 'k', 0, len(times)-1, valinit=k, color='#AAAAAA')
changed_function = partial(slider_changed, data_arrays=data_arrays, plots=plots, fig=fig, times=times, texts=texts)
slider.on_changed(changed_function)

arrow_pressed_function = partial(arrow_pressed, data_arrays=data_arrays, plots=plots, fig=fig, slider=slider, times=times, texts=texts)
fig.canvas.mpl_connect('key_press_event',arrow_pressed_function)

plt.show()

# <codecell>


