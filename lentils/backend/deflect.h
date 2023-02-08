#ifndef _DEFLECT_H_
#define _DEFLECT_H_


#include <stdint.h>


typedef struct {

	unsigned int type;
	unsigned int _pad;
	double z_l, z_s, d_l, d_s, d_ls, sigma_c, beta;
	double fpars[24];
	int ipars[8];
	unsigned char flags[32];

} generic_mass_model;


typedef struct {

	int num_lenses;
	int num_zplanes;
	generic_mass_model *lenses;

} global_lens_model;


void deflect(global_lens_model glm, double *x_in, double *x_out, int want_deriv, double *deriv);


void deflect_points(global_lens_model glm, double *p_in, int npoints, double *p_out, int want_deriv, double *deriv);































#endif // _DEFLECT_H_

