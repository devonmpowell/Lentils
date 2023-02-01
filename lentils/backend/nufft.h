#ifndef _NUFFT_H_
#define _NUFFT_H_

#include "common.h"

#define FORWARD 0 
#define ADJOINT 1 


double kb_g(double argk, double argq, double beta, double wsup);

double kb_w(double argx, double argy, double beta, double wsup);

double kb_optimal_beta(double padfac, double wsup);

void zero_pad(image_space imspace, double *image, image_space padspace, double *padded, int direction);

void grid_cpu(visibility_space visspace, double *vis, fourier_space gspace, double *grid, 
		int wsup, double kb_beta, int direction);


#endif // _NUFFT_H_

