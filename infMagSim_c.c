#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* A C-program for MT19937: Real number version                */
/*   genrand() generates one pseudorandom real number (double) */
/* which is uniformly distributed on [0,1]-interval, for each  */
/* call. sgenrand(seed) set initial values to the working area */
/* of 624 words. Before genrand(), sgenrand(seed) must be      */
/* called once. (seed is any 32-bit integer except for 0).     */
/* Integer generator is obtained by modifying two lines.       */
/*   Coded by Takuji Nishimura, considering the suggestions by */
/* Topher Cooper and Marc Rieffel in July-Aug. 1997.           */

/* This library is free software; you can redistribute it and/or   */
/* modify it under the terms of the GNU Library General Public     */
/* License as published by the Free Software Foundation; either    */
/* version 2 of the License, or (at your option) any later         */
/* version.                                                        */
/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of  */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.            */
/* See the GNU Library General Public License for more details.    */
/* You should have received a copy of the GNU Library General      */
/* Public License along with this library; if not, write to the    */
/* Free Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA   */ 
/* 02111-1307  USA                                                 */

/* Copyright (C) 1997 Makoto Matsumoto and Takuji Nishimura.       */
/* Any feedback is very welcome. For any question, comments,       */
/* see http://www.math.keio.ac.jp/matumoto/emt.html or email       */
/* matumoto@math.keio.ac.jp                                        */

/* Period parameters */  
#define N 624
#define M 397
#define MATRIX_A 0x9908b0df   /* constant vector a */
#define UPPER_MASK 0x80000000 /* most significant w-r bits */
#define LOWER_MASK 0x7fffffff /* least significant r bits */

/* Tempering parameters */   
#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y)  (y >> 11)
#define TEMPERING_SHIFT_S(y)  (y << 7)
#define TEMPERING_SHIFT_T(y)  (y << 15)
#define TEMPERING_SHIFT_L(y)  (y >> 18)

static unsigned long mt[N]; /* the array for the state vector  */
static int mti=N+1; /* mti==N+1 means mt[N] is not initialized */

/* initializing the array with a NONZERO seed */
void
sgenrand(seed)
    unsigned long seed;	
{
    /* setting initial seeds to mt[N] using         */
    /* the generator Line 25 of Table 1 in          */
    /* [KNUTH 1981, The Art of Computer Programming */
    /*    Vol. 2 (2nd Ed.), pp102]                  */
    mt[0]= seed & 0xffffffff;
    for (mti=1; mti<N; mti++)
        mt[mti] = (69069 * mt[mti-1]) & 0xffffffff;
}

float /* generating reals */
/* unsigned long */ /* for integer generation */
genrand()
{
    unsigned long y;
    static unsigned long mag01[2]={0x0, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N+1)   /* if sgenrand() has not been called, */
            sgenrand(4357); /* a default initial seed is used   */

        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (;kk<N-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1];

        mti = 0;
    }
  
    y = mt[mti++];
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);

    return ( (float)y / (unsigned long)0xffffffff ); /* reals */
    /* return y; */ /* for integer generation */
}



/* this main() outputs first 1000 generated numbers  */
/*main()
{ 
    int j;

    sgenrand(4357); any nonzero integer can be used as a seed
for (j=0; j<1000; j++) {
        printf("%5f ", genrand());
        if (j%8==7) printf("\n");
    }
    printf("\n");
} */		   


void move_particles_c(float *grid, float *object_mask, float *potential,
		      float dt, float charge_to_mass, float background_density, int largest_index,
		      float *particles, float *density, int *empty_slots,
		      int *current_empty_slot, float a_b, int update_position, int periodic_particles,
		      int largest_allowed_slot, int n_points, int particle_storage_length) {
  float z_min = grid[0];
  float z_max = grid[n_points-1];
  float dz = (z_max-z_min)/(n_points-1);
  float inactive_slot_position_flag = 2.*z_max;
  float eps=1.e-6;
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
      if (within_bounds) {
	position_offset = fmodf(particles[ix]-z_min,z_max-z_min);
	if (position_offset>=0.) {
	  particles[ix] = position_offset + z_min;
	} else {
	  particles[ix] = position_offset + z_max;
	}
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
      accel = charge_to_mass*electric_field-a_b;
      particles[iv] += accel*dt;
      if (update_position) {
	particles[ix] += particles[iv]*dt;
	if (periodic_particles) {
	  within_bounds = particles[ix]<0.99*inactive_slot_position_flag;
	  if (within_bounds) {
	    position_offset = fmodf(particles[ix]-z_min,z_max-z_min);
	    if (position_offset>=0.) {
	      particles[ix] = position_offset + z_min;
	    } else {
	      particles[ix] = position_offset + z_max;
	    }
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

void move_particles_c_minimal(float *grid, float *object_mask, float *potential,
		      float dt, float charge_to_mass, float background_density, int largest_index,
		      float *particles, float *density, int *empty_slots,
		      int *current_empty_slot, int update_position, int periodic_particles,
		      int largest_allowed_slot, int n_points, int particle_storage_length) {
  float z_min = grid[0];
  float z_max = grid[n_points-1];
  float dz = (z_max-z_min)/(n_points-1);
  float inactive_slot_position_flag = 2.*z_max;
  float eps=1.e-6;
  int j;
  for (j=0; j<n_points; j++)
    density[j] = 0;
  int i;
  for (i=0; i<largest_index+1; i++) {
    int within_bounds, within_bounds_before_move, inside_object;
    int left_index;
    float electric_field;
    float accel;
    float fraction_to_left;
    float position_offset;
    int ix = i;
    int iv = particle_storage_length+i;
    float current_position=particles[ix];
    float current_velocity=particles[iv];
    /*if (periodic_particles) {
      within_bounds = particles[ix]<0.99*inactive_slot_position_flag;
      position_offset = fmodf(particles[ix]-z_min,z_max-z_min);
      if (position_offset>=0.) {
	particles[ix] = position_offset + z_min;
      } else {
	particles[ix] = position_offset + z_max;
      }
    } else {
    */
      within_bounds = (current_position>z_min+eps) && (current_position<z_max-eps);
    //}
    within_bounds_before_move = within_bounds;
    inside_object = 0;
    if (within_bounds_before_move) {
      left_index = (current_position-z_min)/dz;
      /*if (periodic_particles && left_index>n_points-2)
	left_index = 0;
      if (left_index<0 || left_index>n_points-2)
	printf("bad left_index: %d, %f, %f, %f\n", left_index, z_min, particles[ix], z_max);
      */
      electric_field = -(potential[left_index+1]-potential[left_index])/dz;
      accel = charge_to_mass*electric_field;
      current_velocity += accel*dt;
      particles[iv] = current_velocity;
      if (update_position) {
	current_position += current_velocity*dt;
	particles[ix] = current_position;
	/*if (periodic_particles) {
	  within_bounds = particles[ix]<0.99*inactive_slot_position_flag;
	  position_offset = fmodf(particles[ix]-z_min,z_max-z_min);
	  if (position_offset>=0.) {
	    particles[ix] = position_offset + z_min;
	  } else {
	    particles[ix] = position_offset + z_max;
	  }
	} else {
	*/
	  within_bounds = (current_position>z_min+eps) && (current_position<z_max-eps);
	//}
	if (within_bounds) {
	  left_index = (current_position-z_min)/dz;
	  /*if (periodic_particles && left_index>n_points-2)
	    left_index = 0;
	  if (left_index<0 || left_index>n_points-2)
	    printf("bad left_index: %d, %f, %f, %f\n", left_index, z_min, particles[ix], z_max);
	  */
	  if (current_position>0) {
	    fraction_to_left = fmodf(current_position,dz)/dz;
	  } else {
	    fraction_to_left = 1.+fmodf(current_position,dz)/dz;
	  }
	}
      }
    }
    /*if (within_bounds) {
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
    */
    if (within_bounds_before_move) {// || (particles[ix]<0.99*inactive_slot_position_flag)) {
      if (!within_bounds || inside_object) {
	particles[ix] = inactive_slot_position_flag;
	*current_empty_slot += 1;
	if (*current_empty_slot>largest_allowed_slot)
	  printf("bad current_empty_slot: %d, %d\n", *current_empty_slot, largest_allowed_slot);
	empty_slots[*current_empty_slot] = i;
      } else {
	if (current_position>0) {
	  fraction_to_left = fmodf(current_position,dz)/dz;
	} else {
	  fraction_to_left = 1.+fmodf(current_position,dz)/dz;
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
  float z_min = grid[0];
  float z_max = grid[n_points-1];
  float dz = (z_max-z_min)/(n_points-1);
  float k = 1.;
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
  //diagonal[0] = 1.;
  //upper_diagonal[0] = 0.;
  diagonal[0] = dz*dz/debye_length/debye_length*k/2.+dz/debye_length;
  upper_diagonal[0] = dz*dz/debye_length/debye_length*k/2.-dz/debye_length;
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
    //lower_diagonal[n_points-2] = 0.;
    //diagonal[n_points-1] = 1.;
    lower_diagonal[n_points-2] = dz*dz/debye_length/debye_length*k/2.-dz/debye_length;
    diagonal[n_points-1] = dz*dz/debye_length/debye_length*k/2.+dz/debye_length;
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

void histogram2d_uniform_grid_c(float *X, float *Y, float x_min, float step_x, float y_min, float step_y, 
				int n_bins_x, int n_bins_y, int n_data, int *hist)
{
  int index_x;
  int index_y;
  int i;
    for (i = 0; i < n_data; i++)
      {
	index_x = (int)floor((X[i]-x_min)/step_x);
	index_y = (int)floor((Y[i]-y_min)/step_y);
	if (index_x>=0 && index_x<=n_bins_x-1 && index_y>=0 && index_y<=n_bins_y-1)
	  {
	    hist[index_x*n_bins_y+index_y] = hist[index_x*n_bins_y+index_y]+1;
	  }
      }
}

void draw_velocities_c(int n, float v_th, float v_d, float *v_array)
{
  
  int i;
  float half_width;
  float beta;
  float ratio;
  int rejection;
  int n_Iteration;
  float sample_1;
  float sample_2;
  float sample_3;
  float v_max;
  float v_lower;
  float v_upper;
  float v;
  half_width = 5;
  beta = v_d/(sqrt(2)*v_th);
  ratio = (1/sqrt(M_PI)*exp(-pow(beta,2))-beta*(1-erf(beta)))/(1/sqrt(M_PI)*exp(-pow(beta,2))+beta*(1+erf(beta))); 
  /*sgenrand(time(NULL));*/
  for (i=0; i<n; i++)
    {
      rejection = 1;
      n_Iteration = 1;
      sample_1 = genrand()-1/(1+ratio);
      if (sample_1>=0)
	{
	  while (rejection==1)
	    {
	      if (n_Iteration==1000)
		{
		  printf("Warning: method draw_velocities_c doesn't converge");
		  printf("\n");
		  printf("v_max: ");
		  printf("%5f ",v_max);
		  printf("v_upper: ");
		  printf("%5f ",v_upper);
		  printf("v_lower: ");
		  printf("%5f ",v_lower);
		  printf("v_d: ");
		  printf("%5f ",v_d);
		  printf("v_th: ");
		  printf("%5f ",v_th);
		  printf("v: ");
		  printf("%5f ",v);
		  printf("%5f ",(fabs(v)*exp(-pow(v+v_d,2)/(2*pow(v_th,2))))/(fabs(v_max)*exp(-pow(v_max+v_d,2)/(2*pow(v_th,2)))));
		  printf("\n");
		}
	      v_max = (-v_d+sqrt(pow(v_d,2)+4*pow(v_th,2)))/2;
	      v_upper = v_max+half_width*v_th;
	      v_lower = fmax(v_max-half_width*v_th,0.);
	      sample_2 = genrand()*(v_upper-v_lower)+v_lower;
	      v = sample_2;
	      sample_3 = genrand();
	      if (log(sample_3)<=log(fabs(v)/fabs(v_max))+(pow(v_max+v_d,2)-pow(v+v_d,2))/(2*pow(v_th,2)))
		{
		  rejection = 0;
		  v_array[i] = v;
		}
	      else
		{
		  n_Iteration ++;
		}
	    }
	}
      else
	{
	  while (rejection==1)
	    {
	      if (n_Iteration==1000)
		{
		  printf("Warning: method draw_velocities_c doesn't converge");
		  printf("\n");
		  printf("v_max: ");
		  printf("%5f ",v_max);
		  printf("v_upper: ");
		  printf("%5f ",v_upper);
		  printf("v_lower: ");
		  printf("%5f ",v_lower);
		  printf("v_d: ");
		  printf("%5f ",v_d);
		  printf("v_th: ");
		  printf("%5f ",v_th);
		  printf("v: ");
		  printf("%5f ",v);
		  printf("%5f ",(fabs(v)*exp(-pow(v+v_d,2)/(2*pow(v_th,2))))/(fabs(v_max)*exp(-pow(v_max+v_d,2)/(2*pow(v_th,2)))));
		  printf("\n");
		}
	      v_max = (-v_d-sqrt(pow(v_d,2)+4*pow(v_th,2)))/2;
	      v_upper = fmin(v_max+half_width*v_th,0);
	      v_lower = v_max-half_width*v_th;
	      sample_2 = genrand()*(v_upper-v_lower)+v_lower;
	      v = sample_2;
	      sample_3 = genrand();
	      if (log(sample_3)<=log(fabs(v)/fabs(v_max))+(pow(v_max+v_d,2)-pow(v+v_d,2))/(2*pow(v_th,2)))
		{
		  rejection = 0;
		  v_array[i] = v;
		}
	      else
		{
		  n_Iteration ++;
		}
	    }
	}
    }	 
}		    

void electric_field_filter_c(int n_points, float *grid, float *data_array, float L, int n_fine_mesh, float *fine_mesh, float *Res)
{
  
  int i;
  int j;
  float sum;
  for (i=0; i<n_fine_mesh; i++)
  {
    sum = 0;
    for (j=0; j<n_points; j++)
      {
	sum = sum+tanh((fine_mesh[i]-grid[j])/L)/cosh((fine_mesh[i]-grid[j])/L)*data_array[j];
      }
    Res[i] = sum;
  }
}
 
	     
      
  



	
