#include <stdio.h>
#include <math.h>

void move_particles_c(float *grid, float *object_mask, float *potential,
		      float dt, float charge_to_mass, float background_density, int largest_index,
		      float *particles, float *density, int *empty_slots,
		      int *current_empty_slot, int update_position, int periodic_particles,
		      int largest_allowed_slot, int n_points, int particle_storage_length) {
  float z_min = grid[0];
  float z_max = grid[n_points-1];
  float dz = grid[1]-grid[0];
  float inactive_slot_position_flag = 2.*z_max;
  float eps=1.e-5;
  int j;
  for (j=0; j<n_points; j++)
    density[j] = 0;
  int within_bounds, within_bounds_before_move, inside_object;
  int left_index;
  float electric_field;
  float accel;
  float fraction_to_left;
  //float current_potential;
  int i;
  for (i=0; i<largest_index+1; i++) {
    int ix = i;
    int iv = particle_storage_length+i;
    if (periodic_particles) {
      within_bounds = particles[ix]<0.99*inactive_slot_position_flag;
      particles[ix] = fmodf(particles[ix]-z_min,z_max-z_min) + z_min;
    } else {
      within_bounds = (particles[ix]>z_min+eps) && (particles[ix]<z_max-eps);
    }
    within_bounds_before_move = within_bounds;
    inside_object = 0;
    if (within_bounds_before_move) {
      left_index = (particles[ix]-z_min)/dz;
      if (periodic_particles && left_index>n_points-2)
	left_index = 0;
      if (left_index<0 || left_index>n_points-2)
	printf("bad left_index: %d, %f, %f, %f", left_index, z_min, particles[ix], z_max);
      electric_field = -(potential[left_index+1]-potential[left_index])/dz;
      accel = charge_to_mass*electric_field;
      particles[iv] += accel*dt;
      if (update_position) {
	particles[ix] += particles[iv]*dt;
	if (periodic_particles) {
	  within_bounds = particles[ix]<0.99*inactive_slot_position_flag;
	  particles[ix] = fmodf(particles[ix]-z_min,z_max-z_min) + z_min;
	} else {
	  within_bounds = (particles[ix]>z_min+eps) && (particles[ix]<z_max-eps);
	}
	if (within_bounds) {
	  left_index = (particles[ix]-z_min)/dz;
	  if (periodic_particles && left_index>n_points-2)
	    left_index = 0;
	  if (left_index<0 || left_index>n_points-2)
	    printf("bad left_index: %d, %f, %f, %f", left_index, z_min, particles[ix], z_max);
	  if (particles[ix]>0) {
	    fraction_to_left = fmodf(particles[ix],dz)/dz;
	  } else {
	    fraction_to_left = 1.+fmodf(particles[ix],dz)/dz;
	  }
	  //current_potential = potential[left_index]*fraction_to_left +
	  //  potential[left_index+1]*(1.-fraction_to_left);
	}
      }
    }
    if (within_bounds) {
      if (object_mask[left_index]>0.) {
	if (object_mask[left_index+1]>0.) {
	  if (particles[ix]>grid[left_index]+(1.-object_mask[left_index])*dz)
	    inside_object = 1;
	} else {
	  if (particles[ix]>grid[left_index]+object_mask[left_index]*dz)
	    inside_object = 1;
	}
      }
    }
    if (within_bounds_before_move || (particles[ix]<0.99*inactive_slot_position_flag)) {
      if (!within_bounds || inside_object) {
	particles[ix] = inactive_slot_position_flag;
	*current_empty_slot += 1;
	if (*current_empty_slot>largest_allowed_slot)
	  printf("bad current_empty_slot: %d, %d", *current_empty_slot, largest_allowed_slot);
	empty_slots[*current_empty_slot] = i;
      } else {
	if (particles[ix]>0) {
	  fraction_to_left = fmodf(particles[ix],dz)/dz;
	} else {
	  fraction_to_left = 1.+fmodf(particles[ix],dz)/dz;
	}
	density[left_index] += fraction_to_left/background_density;
	density[left_index+1] += (1-fraction_to_left)/background_density;
      }
    }
  }
}
