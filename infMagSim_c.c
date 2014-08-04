#include <stdio.h>
#include <stdlib.h>
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
  float position_offset;
  //float current_potential;
  int i;
  for (i=0; i<largest_index+1; i++) {
    int ix = i;
    int iv = particle_storage_length+i;
    if (periodic_particles) {
      within_bounds = particles[ix]<0.99*inactive_slot_position_flag;
      position_offset = fmodf(particles[ix]-z_min,z_max-z_min);
      if (position_offset>=0.) {
	particles[ix] = position_offset + z_min;
      } else {
	particles[ix] = position_offset + z_max;
      }
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
	printf("bad left_index: %d, %f, %f, %f\n", left_index, z_min, particles[ix], z_max);
      electric_field = -(potential[left_index+1]-potential[left_index])/dz;
      accel = charge_to_mass*electric_field;
      particles[iv] += accel*dt;
      if (update_position) {
	particles[ix] += particles[iv]*dt;
	if (periodic_particles) {
	  within_bounds = particles[ix]<0.99*inactive_slot_position_flag;
	  position_offset = fmodf(particles[ix]-z_min,z_max-z_min);
	  if (position_offset>=0.) {
	    particles[ix] = position_offset + z_min;
	  } else {
	    particles[ix] = position_offset + z_max;
	  }
	} else {
	  within_bounds = (particles[ix]>z_min+eps) && (particles[ix]<z_max-eps);
	}
	if (within_bounds) {
	  left_index = (particles[ix]-z_min)/dz;
	  if (periodic_particles && left_index>n_points-2)
	    left_index = 0;
	  if (left_index<0 || left_index>n_points-2)
	    printf("bad left_index: %d, %f, %f, %f\n", left_index, z_min, particles[ix], z_max);
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
	  printf("bad current_empty_slot: %d, %d\n", *current_empty_slot, largest_allowed_slot);
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

void tridiagonal_solve_c(double *a, double *b, double *c, double *d, double *x, int n) { // also modifies b and d
  //first element of a is a row below first elements of b, c, and d
  int j;
  for (j=0; j<n-1; j++) {
    d[j+1] -= d[j] * a[j] / b[j];
    b[j+1] -= c[j] * a[j] / b[j];
  }
  for (j=n-2; j>=0; j--)
    d[j] -= d[j+1] * c[j] / b[j+1];
  for (j=0; j<n; j++)
    x[j] = d[j]/b[j];
}

void poisson_solve_c(float *grid, float *object_mask, float *charge, float debye_length,
		     float *potential, float object_potential, float object_transparency,
		     int boltzmann_electrons, int periodic_potential, int n_points) {
  // if boltzmann_electrons then charge=ion_charge
  float eps = 1.e-5;
  //float z_min = grid[0];
  //float z_max = grid[n_points-1];
  float dz = grid[1]-grid[0];
  int j;

  double *diagonal, *lower_diagonal, *upper_diagonal, *right_hand_side;
  diagonal = (double*)malloc(n_points*sizeof(double));
  lower_diagonal = (double*)malloc(n_points*sizeof(double));
  upper_diagonal = (double*)malloc(n_points*sizeof(double));
  right_hand_side = (double*)malloc(n_points*sizeof(double));

  double *diagonal_obj, *lower_diagonal_obj, *upper_diagonal_obj, *right_hand_side_obj;
  diagonal_obj = (double*)malloc(n_points*sizeof(double));
  lower_diagonal_obj = (double*)malloc(n_points*sizeof(double));
  upper_diagonal_obj = (double*)malloc(n_points*sizeof(double));
  right_hand_side_obj = (double*)malloc(n_points*sizeof(double));

  double *periodic_u, *periodic_v, *periodic_w, *result;
  periodic_u = (double*)malloc(n_points*sizeof(double));
  periodic_v = (double*)malloc(n_points*sizeof(double));
  periodic_w = (double*)malloc(n_points*sizeof(double));
  result = (double*)malloc(n_points*sizeof(double));

  for (j=0; j<n_points; j++) {
    diagonal[j] = -2.;
    lower_diagonal[j] = 1.;
    upper_diagonal[j] = 1.;
    right_hand_side[j] = charge[j]; // scale later
    periodic_u[j] = 0.;
    periodic_v[j] = 0.;
    periodic_w[j] = 0.;
    result[j] = 0.;
  }
  right_hand_side[n_points-1] = 0.;
  if (boltzmann_electrons) {
    for (j=0; j<n_points; j++) {
      diagonal[j] -= dz*dz*exp(potential[j])/debye_length/debye_length;
      right_hand_side[j] -= exp(potential[j])*(1.-potential[j]);
    }
  }
  for (j=0; j<n_points; j++)
    right_hand_side[j] *= -dz*dz/debye_length/debye_length;
  // Dirichlet left boundary
  diagonal[0] = 1.;
  upper_diagonal[0] = 0.;
  right_hand_side[0] = 0.;
  for (j=0; j<n_points; j++) {
    periodic_u[j] = 0.;
    periodic_v[j] = 0.;
  }
  if (periodic_potential) {
    // Enforce last element equal first
    lower_diagonal[n_points-2] = 0.;
    diagonal[n_points-1] = 1.;
    right_hand_side[n_points-1] = 0.;
    // Use Sherman-Morrison formula for non-tidiagonal part
    periodic_u[n_points-1] = 1.;
    periodic_v[0] = -1.;
  } else {
    // Dirichlet right boundary
    lower_diagonal[n_points-2] = 0.;
    diagonal[n_points-1] = 1.;
    right_hand_side[n_points-1] = 0.;
  }
  for (j=0; j<n_points; j++) {
    diagonal_obj[j] = diagonal[j];
    lower_diagonal_obj[j] = lower_diagonal[j];
    upper_diagonal_obj[j] = upper_diagonal[j];
    right_hand_side_obj[j] = right_hand_side[j];
  }
  // TODO: check that object solve works properly with Boltzmann electrons

  // TODO: resolve issues when object boundaries fall on grid points
  float zeta_left, zeta_right;
  for (j=1; j<n_points-1; j++) {
    if (object_mask[j]>0.) {
      if (object_mask[j]<1.-eps) {
	// Partially covered cell
	if (object_mask[j+1]>0.) {
	  // left closest point outside
	  zeta_left = (1.-object_mask[j])*dz;
	  lower_diagonal_obj[j-1] = zeta_left/dz; // -1 since lower diagonal shifted
	  diagonal_obj[j] = -1.-zeta_left/dz;
	  upper_diagonal_obj[j] = 0.;
	  right_hand_side_obj[j] *= zeta_left/dz;
	  right_hand_side_obj[j] -= object_potential;
	} else if (object_mask[j-1]<1.-eps) {
	  // Only point inside
	  zeta_left = (1.-object_mask[j-1])*dz;
	  zeta_right = (1.-object_mask[j])*dz;
	  lower_diagonal_obj[j-1] = -(1.-zeta_left/dz); // -1 since lower diagonal shifted
	  diagonal_obj[j] = -(zeta_left+zeta_right)/dz;
	  upper_diagonal_obj[j] = -(1.-zeta_right/dz);
	  right_hand_side_obj[j] = -2.*object_potential;
	} else if (object_mask[j-1]>1.-eps) {
	  // right closest point inside
	  zeta_right = (1.-object_mask[j])*dz;
	  lower_diagonal_obj[j-1] = 0.; // -1 since lower diagonal shifted
	  diagonal_obj[j] = -2.*zeta_right/dz;
	  upper_diagonal_obj[j] = -2.*(1.-zeta_right/dz);
	  right_hand_side_obj[j] = -2.*object_potential;
	} else {
	  printf("bad object_mask1\n");
	}
      } else if (object_mask[j]>1.-eps) {
	if (object_mask[j-1]<1.-eps) {
	  // left closest point inside
	  zeta_left = (1.-object_mask[j-1])*dz;
	  lower_diagonal_obj[j-1] = -2.*(1.-zeta_left/dz); // -1 since lower diagonal shifted
	  diagonal_obj[j] = -2.*zeta_left/dz;
	  upper_diagonal_obj[j] = 0.;
	  right_hand_side_obj[j] = -2.*object_potential;
	} else if (object_mask[j-1]>1.-eps) {
	  // Interior point
	  right_hand_side_obj[j] = 0.; // Constant derivative inside object (hopefully zero)
	} else {
	  printf("bad object_mask2\n");
	}
      } else {
	printf("bad object_mask3\n");
      }
    } else if (object_mask[j-1]>0.) {
      // right closest point outside
      zeta_right = (1.-object_mask[j-1])*dz;
      if (j>0)
	lower_diagonal_obj[j-1] = 0.; // -1 since lower diagonal shifted
      diagonal_obj[j] = -1.-zeta_right/dz;
      upper_diagonal_obj[j] = zeta_right/dz;
      right_hand_side_obj[j] *= zeta_right/dz;
      right_hand_side_obj[j] -= object_potential;
    }
  }
  for (j=0; j<n_points; j++) {
    upper_diagonal[j] = upper_diagonal[j]*object_transparency + upper_diagonal_obj[j]*(1.-object_transparency);
    diagonal[j] = diagonal[j]*object_transparency + diagonal_obj[j]*(1.-object_transparency);
    lower_diagonal[j] = lower_diagonal[j]*object_transparency + lower_diagonal_obj[j]*(1.-object_transparency);
    right_hand_side[j] = right_hand_side[j]*object_transparency + right_hand_side_obj[j]*(1.-object_transparency);
  }
  if (periodic_potential) {
    // Use Sherman-Morrison formula to deal with non-tridiagonal part
    tridiagonal_solve_c(lower_diagonal, diagonal, upper_diagonal, right_hand_side, result, n_points);
    tridiagonal_solve_c(lower_diagonal, diagonal, upper_diagonal, periodic_u, periodic_w, n_points);
    double v_dot_result=0.;
    double v_dot_w=0.;
    for (j=0; j<n_points; j++)
      v_dot_result += periodic_v[j]*result[j];
    v_dot_w += periodic_v[j]*periodic_w[j];
    for (j=0; j<n_points; j++)
      result[j] = result[j] - v_dot_result/(1.+v_dot_w)*periodic_w[j];
  } else {
    tridiagonal_solve_c(lower_diagonal, diagonal, upper_diagonal, right_hand_side, result, n_points);
  }
  double average_potential = 0.;
  if (periodic_potential) {
    for (j=0; j<n_points; j++)
      average_potential += result[j];
    average_potential /= n_points;
  }
  for (j=0; j<n_points; j++)
    potential[j] = result[j]-average_potential;

  free(diagonal);
  free(lower_diagonal);
  free(upper_diagonal);
  free(right_hand_side);

  free(diagonal_obj);
  free(lower_diagonal_obj);
  free(upper_diagonal_obj);
  free(right_hand_side_obj);

  free(periodic_u);
  free(periodic_v);
  free(periodic_w);
  free(result);
}
