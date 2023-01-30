#ifndef _DEFLECT_H_
#define _DEFLECT_H_


#include <stdint.h>


typedef struct {

	unsigned long type;
	double z_l, z_s, d_l, d_s, d_ls, sigma_c, beta;
	double fpars[32];
	unsigned char flags[32];

} generic_mass_model;




































#endif // _DEFLECT_H_

